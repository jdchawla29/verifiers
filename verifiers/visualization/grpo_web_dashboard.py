"""
Comprehensive web dashboard for GRPO training with real-time pipeline tracking.
Shows the exact state of batches moving through generation, scoring, and training.
"""

import json
import time
import threading
import asyncio
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Deque
import logging

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from transformers import TrainerCallback
import numpy as np

from .telemetry_db import TelemetryDatabase, RolloutInfo

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """Information about a batch in the pipeline."""
    batch_id: int
    status: str  # 'queued', 'generating', 'generated', 'scoring', 'scored', 'training', 'completed'
    created_at: float
    num_samples: int
    num_generations: int
    
    # Timing
    generation_start: Optional[float] = None
    generation_end: Optional[float] = None
    scoring_start: Optional[float] = None
    scoring_end: Optional[float] = None
    training_start: Optional[float] = None
    training_end: Optional[float] = None
    
    # Data
    prompts: Optional[List[str]] = None
    generations: Optional[List[List[str]]] = None  # [sample][generation]
    scores: Optional[List[List[float]]] = None
    rewards: Optional[List[float]] = None
    advantages: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert times to relative for display
        now = time.time()
        if self.created_at:
            d['age'] = now - self.created_at
        return d


class GRPOPipelineTracker:
    """Tracks the exact state of the GRPO training pipeline."""
    
    def __init__(self, db_path: str = "grpo_telemetry.db"):
        self.batches: Dict[int, BatchInfo] = {}
        self.generation_queue: Deque[int] = deque()
        self.scoring_queue: Deque[int] = deque()
        self.training_queue: Deque[int] = deque()
        self.completed: Deque[int] = deque(maxlen=20)
        
        self.metrics = defaultdict(lambda: deque(maxlen=100))
        self.current_step = 0
        self.total_steps = 0
        self.next_batch_id = 0
        
        self.lock = threading.Lock()
        
        # Initialize telemetry database
        self.db = TelemetryDatabase(db_path)
        
        # Track rollout-level information
        self.rollouts: Dict[str, RolloutInfo] = {}  # rollout_id -> RolloutInfo
        self.rollout_timings: Dict[str, Dict[str, float]] = {}  # For async tracking
        
    def create_batch(self, num_samples: int, num_generations: int) -> int:
        """Create a new batch and add to generation queue."""
        with self.lock:
            batch_id = self.next_batch_id
            self.next_batch_id += 1
            
            batch = BatchInfo(
                batch_id=batch_id,
                status='queued',
                created_at=time.time(),
                num_samples=num_samples,
                num_generations=num_generations
            )
            
            self.batches[batch_id] = batch
            self.generation_queue.append(batch_id)
            
            # Create batch in database
            self.db.create_batch(batch_id, num_samples, num_generations)
            
            # Create rollout entries
            for prompt_idx in range(num_samples):
                for gen_idx in range(num_generations):
                    rollout_id = f"{batch_id}:{prompt_idx}:{gen_idx}"
                    rollout_info = RolloutInfo(
                        rollout_id=rollout_id,
                        batch_id=batch_id,
                        prompt_idx=prompt_idx,
                        generation_idx=gen_idx,
                        prompt=f"Prompt {prompt_idx}",  # Will be updated later
                        status='pending'
                    )
                    self.rollouts[rollout_id] = rollout_info
                    self.db.create_rollout(rollout_info)
            
            return batch_id
            
    def start_generation(self, batch_id: int) -> None:
        """Mark batch as starting generation."""
        with self.lock:
            if batch_id in self.batches:
                start_time = time.time()
                self.batches[batch_id].status = 'generating'
                self.batches[batch_id].generation_start = start_time
                
                # Remove from queue
                if batch_id in self.generation_queue:
                    self.generation_queue.remove(batch_id)
                
                # Update database
                self.db.update_batch_status(batch_id, 'generating', generation_start=start_time)
                
                # Track per-rollout generation start
                for rollout_id in self.rollouts:
                    if rollout_id.startswith(f"{batch_id}:"):
                        self.rollout_timings[rollout_id] = {'generation_start': start_time}
                    
    def complete_generation(self, batch_id: int, prompts: List[str], 
                          generations: List[List[str]]) -> None:
        """Mark generation complete with results."""
        with self.lock:
            if batch_id in self.batches:
                end_time = time.time()
                batch = self.batches[batch_id]
                batch.status = 'generated'
                batch.generation_end = end_time
                batch.prompts = prompts[:3]  # Store first 3 for display
                batch.generations = [gens[:3] for gens in generations[:3]]  # First 3 samples, 3 gens each
                
                # Move to scoring queue
                self.scoring_queue.append(batch_id)
                
                # Update database
                self.db.update_batch_status(batch_id, 'generated', generation_end=end_time)
                
                # Update rollouts with actual prompts and completions
                for prompt_idx, prompt in enumerate(prompts):
                    for gen_idx, completion in enumerate(generations[prompt_idx] if prompt_idx < len(generations) else []):
                        rollout_id = f"{batch_id}:{prompt_idx}:{gen_idx}"
                        if rollout_id in self.rollouts:
                            self.rollouts[rollout_id].prompt = prompt
                            self.rollouts[rollout_id].completion = completion
                            
                            # Update database with completion
                            if rollout_id in self.rollout_timings:
                                start_time = self.rollout_timings[rollout_id].get('generation_start', end_time)
                                # Estimate tokens (rough approximation)
                                tokens = len(completion.split()) * 1.3
                                self.db.update_rollout_generation(
                                    rollout_id, completion, start_time, end_time,
                                    int(tokens)
                                )
                
    def start_scoring(self, batch_id: int) -> None:
        """Mark batch as starting scoring."""
        with self.lock:
            if batch_id in self.batches:
                start_time = time.time()
                self.batches[batch_id].status = 'scoring'
                self.batches[batch_id].scoring_start = start_time
                
                if batch_id in self.scoring_queue:
                    self.scoring_queue.remove(batch_id)
                
                # Update database
                self.db.update_batch_status(batch_id, 'scoring', scoring_start=start_time)
                
                # Track per-rollout scoring start
                for rollout_id in self.rollouts:
                    if rollout_id.startswith(f"{batch_id}:"):
                        if rollout_id in self.rollout_timings:
                            self.rollout_timings[rollout_id]['scoring_start'] = start_time
                    
    def complete_scoring(self, batch_id: int, rewards: List[float], 
                        advantages: List[float], scores: List[List[float]]) -> None:
        """Mark scoring complete with results."""
        with self.lock:
            if batch_id in self.batches:
                end_time = time.time()
                batch = self.batches[batch_id]
                batch.status = 'scored'
                batch.scoring_end = end_time
                batch.rewards = rewards
                batch.advantages = advantages
                batch.scores = [s[:3] for s in scores[:3]]  # Match stored generations
                
                # Move to training queue
                self.training_queue.append(batch_id)
                
                # Update database
                self.db.update_batch_status(batch_id, 'scored', scoring_end=end_time)
                
                # Update rollouts with scores
                num_prompts = len(batch.prompts) if batch.prompts else 0
                num_gens = len(batch.generations[0]) if batch.generations and batch.generations[0] else 1
                
                idx = 0
                for prompt_idx in range(num_prompts):
                    for gen_idx in range(num_gens):
                        if idx < len(rewards):
                            rollout_id = f"{batch_id}:{prompt_idx}:{gen_idx}"
                            if rollout_id in self.rollouts:
                                self.rollouts[rollout_id].reward = rewards[idx]
                                self.rollouts[rollout_id].advantage = advantages[idx] if idx < len(advantages) else 0
                                
                                # Extract scores for this rollout
                                rollout_scores = {}
                                if idx < len(scores) and isinstance(scores[idx], list):
                                    for i, score in enumerate(scores[idx]):
                                        rollout_scores[f'score_{i}'] = score
                                
                                # Update database
                                if rollout_id in self.rollout_timings:
                                    start_time = self.rollout_timings[rollout_id].get('scoring_start', end_time)
                                    self.db.update_rollout_scoring(
                                        rollout_id, rewards[idx], 
                                        advantages[idx] if idx < len(advantages) else 0,
                                        rollout_scores, start_time, end_time
                                    )
                            idx += 1
    
    def complete_generation_with_timing(self, batch_id: int, prompts: List[str], 
                                      generations: List[List[str]], 
                                      start_time: float, end_time: float) -> None:
        """Mark generation complete with detailed timing information."""
        # Call the regular method which already handles database updates
        self.complete_generation(batch_id, prompts, generations)
        
        # The timing is already handled in complete_generation via rollout_timings
                
    def start_training(self, batch_id: int) -> None:
        """Mark batch as starting training."""
        with self.lock:
            if batch_id in self.batches:
                self.batches[batch_id].status = 'training'
                self.batches[batch_id].training_start = time.time()
                
                if batch_id in self.training_queue:
                    self.training_queue.remove(batch_id)
                    
    def complete_training(self, batch_id: int, loss: Optional[float] = None) -> None:
        """Mark training complete."""
        with self.lock:
            if batch_id in self.batches:
                end_time = time.time()
                batch = self.batches[batch_id]
                batch.status = 'completed'
                batch.training_end = end_time
                
                # Move to completed
                self.completed.append(batch_id)
                
                # Update database
                self.db.update_batch_status(
                    batch_id, 'completed', 
                    training_end=end_time,
                    completed_at=end_time
                )
                
                # Update rollouts with training loss if provided
                if loss is not None:
                    for rollout_id in self.rollouts:
                        if rollout_id.startswith(f"{batch_id}:"):
                            self.db.update_rollout_training(rollout_id, loss)
                
                # Clean up old batches
                if batch_id in self.batches and len(self.completed) > 20:
                    oldest = self.completed[0]
                    if oldest in self.batches:
                        del self.batches[oldest]
                        
    def update_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Update training metrics."""
        with self.lock:
            self.current_step = step
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics[key].append(value)
                    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state for dashboard."""
        with self.lock:
            # Get batches in each stage
            generating = [bid for bid, b in self.batches.items() if b.status == 'generating']
            scoring = [bid for bid, b in self.batches.items() if b.status == 'scoring']
            training = [bid for bid, b in self.batches.items() if b.status == 'training']
            
            state = {
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'pipeline': {
                    'generation_queue': list(self.generation_queue)[-5:],
                    'generating': generating,
                    'scoring_queue': list(self.scoring_queue)[-5:],
                    'scoring': scoring,
                    'training_queue': list(self.training_queue)[-5:],
                    'training': training,
                    'completed': list(self.completed)[-5:],
                },
                'batches': {bid: b.to_dict() for bid, b in self.batches.items()},
                'metrics': {
                    key: {
                        'current': values[-1] if values else 0,
                        'history': list(values)[-20:],  # Last 20 points
                    }
                    for key, values in self.metrics.items()
                    if key in ['loss', 'reward', 'learning_rate']
                },
                'timestamp': time.time(),
            }
            
            return state


# Global tracker instance
pipeline_tracker = GRPOPipelineTracker()


# Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>GRPO Pipeline Dashboard</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 10px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .pipeline-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }
        .pipeline-stage {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            min-height: 200px;
        }
        .pipeline-stage h3 {
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .batch-card {
            background: #3a3a3a;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }
        .batch-card.generating { border-left-color: #2196F3; background: #1e3a5f; }
        .batch-card.scoring { border-left-color: #FF9800; background: #5f3e1e; }
        .batch-card.training { border-left-color: #9C27B0; background: #3e1e5f; }
        .batch-card.queued { border-left-color: #607D8B; opacity: 0.7; }
        
        .metrics-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        .metrics-cards {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .chart-container {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            height: 300px;
        }
        .examples-container {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .example {
            background: #3a3a3a;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .generation {
            background: #4a4a4a;
            padding: 5px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 0.85em;
        }
        .telemetry-container {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        .telemetry-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab-button {
            padding: 8px 16px;
            background: #2a2a2a;
            border: none;
            color: #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-button:hover {
            background: #333;
        }
        .tab-button.active {
            background: #4CAF50;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .rollout-row {
            display: grid;
            grid-template-columns: 80px 150px 1fr 100px 100px 100px;
            padding: 8px;
            border-bottom: 1px solid #333;
            font-size: 12px;
            align-items: center;
        }
        .rollout-row.header {
            font-weight: bold;
            background: #2a2a2a;
        }
        .generation.best {
            border: 1px solid #4CAF50;
            background: #2e4a2e;
        }
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .status-active { background: #4CAF50; }
        .progress-bar {
            width: 200px;
            height: 20px;
            background: #3a3a3a;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }
        .timing {
            font-size: 0.8em;
            color: #888;
        }
        .spinner {
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0;">🚀 GRPO Pipeline Dashboard</h1>
                <span class="status-indicator status-active"></span>
                <span id="status">Connected</span>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div>Step: <span id="current-step">0</span> / <span id="total-steps">0</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="pipeline-container">
            <div class="pipeline-stage">
                <h3>🚀 Generation <span id="gen-count" style="color: #2196F3;"></span></h3>
                <div id="generation-batches"></div>
            </div>
            <div class="pipeline-stage">
                <h3>📊 Scoring <span id="score-count" style="color: #FF9800;"></span></h3>
                <div id="scoring-batches"></div>
            </div>
            <div class="pipeline-stage">
                <h3>🎯 Training <span id="train-count" style="color: #9C27B0;"></span></h3>
                <div id="training-batches"></div>
            </div>
        </div>
        
        <div class="metrics-container">
            <div class="metrics-cards">
                <div class="metric-card">
                    <div>Loss</div>
                    <div class="metric-value" id="loss-value">-</div>
                </div>
                <div class="metric-card">
                    <div>Reward</div>
                    <div class="metric-value" id="reward-value">-</div>
                </div>
                <div class="metric-card">
                    <div>Queue Depth</div>
                    <div class="metric-value" id="queue-depth">0</div>
                </div>
                <div class="metric-card">
                    <div>Throughput</div>
                    <div class="metric-value" id="throughput">-</div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="metrics-chart"></canvas>
            </div>
        </div>
        
        <div class="examples-container">
            <h3>📝 Recent Rollouts</h3>
            <div id="examples-list"></div>
        </div>
        
        <div class="telemetry-container">
            <h3>📊 Batch Details <span id="selected-batch-id" style="color: #4CAF50;"></span></h3>
            <div class="telemetry-tabs">
                <button class="tab-button active" onclick="showTab('timeline')">Timeline</button>
                <button class="tab-button" onclick="showTab('rollouts')">Rollouts</button>
                <button class="tab-button" onclick="showTab('performance')">Performance</button>
            </div>
            <div id="timeline-tab" class="tab-content active">
                <canvas id="timeline-chart" height="200"></canvas>
            </div>
            <div id="rollouts-tab" class="tab-content">
                <div id="rollouts-table"></div>
            </div>
            <div id="performance-tab" class="tab-content">
                <div id="performance-stats"></div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let chart = null;
        let lastUpdate = Date.now();
        let completedBatches = 0;
        
        // Initialize chart
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: '#f44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y-loss',
                    },
                    {
                        label: 'Reward',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y-reward',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#888' },
                        grid: { color: '#333' }
                    },
                    'y-loss': {
                        type: 'linear',
                        position: 'left',
                        ticks: { color: '#f44336' },
                        grid: { color: '#333' }
                    },
                    'y-reward': {
                        type: 'linear',
                        position: 'right',
                        ticks: { color: '#4CAF50' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
        
        function formatTime(seconds) {
            if (!seconds) return '-';
            return seconds.toFixed(1) + 's';
        }
        
        function updatePipeline(state) {
            // Update progress
            const progress = state.total_steps > 0 ? 
                (state.current_step / state.total_steps * 100) : 0;
            document.getElementById('progress').style.width = progress + '%';
            document.getElementById('current-step').textContent = state.current_step;
            document.getElementById('total-steps').textContent = state.total_steps;
            
            // Update pipeline stages
            const pipeline = state.pipeline;
            const batches = state.batches;
            
            // Generation
            let genHtml = '';
            let genCount = 0;
            
            // Currently generating
            pipeline.generating.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    genHtml += `<div class="batch-card generating">
                        <strong>Batch ${bid}</strong> <span class="spinner">⚡</span><br>
                        ${batch.num_samples}×${batch.num_generations} generations<br>
                        <span class="timing">${formatTime(Date.now()/1000 - batch.generation_start)}</span>
                    </div>`;
                    genCount++;
                }
            });
            
            // Queued
            pipeline.generation_queue.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    genHtml += `<div class="batch-card queued">
                        <strong>Batch ${bid}</strong> ⏳<br>
                        ${batch.num_samples}×${batch.num_generations} generations
                    </div>`;
                    genCount++;
                }
            });
            
            document.getElementById('generation-batches').innerHTML = genHtml || '<div style="color: #666;">Empty</div>';
            document.getElementById('gen-count').textContent = genCount > 0 ? `(${genCount})` : '';
            
            // Scoring
            let scoreHtml = '';
            let scoreCount = 0;
            
            pipeline.scoring.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    scoreHtml += `<div class="batch-card scoring">
                        <strong>Batch ${bid}</strong> <span class="spinner">🔍</span><br>
                        Scoring ${batch.num_samples * batch.num_generations} outputs<br>
                        <span class="timing">${formatTime(Date.now()/1000 - batch.scoring_start)}</span>
                    </div>`;
                    scoreCount++;
                }
            });
            
            pipeline.scoring_queue.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    scoreHtml += `<div class="batch-card queued">
                        <strong>Batch ${bid}</strong> ⏳<br>
                        Gen time: ${formatTime(batch.generation_end - batch.generation_start)}
                    </div>`;
                    scoreCount++;
                }
            });
            
            document.getElementById('scoring-batches').innerHTML = scoreHtml || '<div style="color: #666;">Empty</div>';
            document.getElementById('score-count').textContent = scoreCount > 0 ? `(${scoreCount})` : '';
            
            // Training
            let trainHtml = '';
            let trainCount = 0;
            
            pipeline.training.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    const avgReward = batch.rewards ? 
                        (batch.rewards.reduce((a,b) => a+b, 0) / batch.rewards.length).toFixed(3) : '-';
                    trainHtml += `<div class="batch-card training">
                        <strong>Batch ${bid}</strong> <span class="spinner">⚡</span><br>
                        Avg reward: ${avgReward}<br>
                        <span class="timing">${formatTime(Date.now()/1000 - batch.training_start)}</span>
                    </div>`;
                    trainCount++;
                }
            });
            
            pipeline.training_queue.forEach(bid => {
                const batch = batches[bid];
                if (batch) {
                    const avgReward = batch.rewards ? 
                        (batch.rewards.reduce((a,b) => a+b, 0) / batch.rewards.length).toFixed(3) : '-';
                    trainHtml += `<div class="batch-card queued">
                        <strong>Batch ${bid}</strong> ✅<br>
                        Ready (reward: ${avgReward})
                    </div>`;
                    trainCount++;
                }
            });
            
            document.getElementById('training-batches').innerHTML = trainHtml || '<div style="color: #666;">Empty</div>';
            document.getElementById('train-count').textContent = trainCount > 0 ? `(${trainCount})` : '';
            
            // Update metrics
            if (state.metrics.loss && state.metrics.loss.current !== undefined) {
                document.getElementById('loss-value').textContent = state.metrics.loss.current.toFixed(4);
            }
            if (state.metrics.reward && state.metrics.reward.current !== undefined) {
                document.getElementById('reward-value').textContent = state.metrics.reward.current.toFixed(4);
            }
            
            // Queue depth
            const queueDepth = genCount + scoreCount + trainCount;
            document.getElementById('queue-depth').textContent = queueDepth;
            
            // Throughput
            const newCompleted = pipeline.completed.length;
            if (newCompleted > completedBatches) {
                const elapsed = (Date.now() - lastUpdate) / 1000;
                const rate = (newCompleted - completedBatches) / elapsed;
                document.getElementById('throughput').textContent = rate.toFixed(1) + '/s';
                completedBatches = newCompleted;
                lastUpdate = Date.now();
            }
            
            // Update chart
            if (state.metrics.loss && state.metrics.loss.history) {
                const steps = state.metrics.loss.history.map((_, i) => i);
                chart.data.labels = steps;
                chart.data.datasets[0].data = state.metrics.loss.history;
                if (state.metrics.reward && state.metrics.reward.history) {
                    chart.data.datasets[1].data = state.metrics.reward.history;
                }
                chart.update('none');
            }
            
            // Update examples
            updateExamples(batches, pipeline.completed.slice(-3));
        }
        
        function updateExamples(batches, recentBatchIds) {
            let html = '';
            
            recentBatchIds.reverse().forEach(bid => {
                const batch = batches[bid];
                if (batch && batch.prompts && batch.generations) {
                    html += `<div class="example" data-batch-id="${bid}" style="cursor: pointer;">
                        <strong>Batch ${bid}</strong> - 
                        Total time: ${formatTime(batch.training_end - batch.created_at)}
                        <span style="float: right; color: #4CAF50; font-size: 0.9em;">Click for details →</span><br>`;
                    
                    // Show first example
                    if (batch.prompts[0]) {
                        html += `<div style="margin-top: 10px;">
                            <em>Q: ${batch.prompts[0].substring(0, 100)}...</em><br>`;
                        
                        if (batch.generations[0] && batch.scores && batch.scores[0]) {
                            const bestIdx = batch.scores[0].indexOf(Math.max(...batch.scores[0]));
                            batch.generations[0].forEach((gen, idx) => {
                                const score = batch.scores[0][idx] || 0;
                                const isBest = idx === bestIdx;
                                html += `<div class="generation ${isBest ? 'best' : ''}">
                                    Gen ${idx+1} (score: ${score.toFixed(3)}): 
                                    ${gen.substring(0, 100)}...
                                </div>`;
                            });
                        }
                        html += '</div>';
                    }
                    html += '</div>';
                }
            });
            
            document.getElementById('examples-list').innerHTML = html || '<div style="color: #666;">No examples yet</div>';
        }
        
        // Socket event handlers
        socket.on('connect', () => {
            document.getElementById('status').textContent = 'Connected';
            socket.emit('request_update');
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status').textContent = 'Disconnected';
        });
        
        socket.on('pipeline_update', (data) => {
            updatePipeline(data);
        });
        
        // Request updates
        setInterval(() => {
            socket.emit('request_update');
        }, 500);  // 2 updates per second
        
        // Telemetry functions
        let selectedBatchId = null;
        let timelineChart = null;
        
        function showTab(tabName) {
            // Update buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Load tab data
            if (selectedBatchId) {
                if (tabName === 'timeline') {
                    loadTimeline(selectedBatchId);
                } else if (tabName === 'rollouts') {
                    loadRollouts(selectedBatchId);
                } else if (tabName === 'performance') {
                    loadPerformance();
                }
            }
        }
        
        function selectBatch(batchId) {
            selectedBatchId = batchId;
            document.getElementById('selected-batch-id').textContent = `#${batchId}`;
            
            // Load default tab
            loadTimeline(batchId);
        }
        
        async function loadTimeline(batchId) {
            const response = await fetch(`/api/batch/${batchId}/timeline`);
            const data = await response.json();
            
            if (data.timeline && data.timeline.length > 0) {
                drawTimeline(data.timeline);
            }
        }
        
        async function loadRollouts(batchId) {
            const response = await fetch(`/api/batch/${batchId}`);
            const data = await response.json();
            
            if (data.rollouts) {
                drawRolloutsTable(data.rollouts);
            }
        }
        
        async function loadPerformance() {
            const response = await fetch('/api/performance?window_minutes=10');
            const data = await response.json();
            
            drawPerformanceStats(data);
        }
        
        function drawTimeline(events) {
            const canvas = document.getElementById('timeline-chart');
            const ctx = canvas.getContext('2d');
            
            // Calculate time bounds
            const minTime = Math.min(...events.map(e => e.time));
            const maxTime = Math.max(...events.map(e => e.time));
            const duration = maxTime - minTime;
            
            // Group events by rollout
            const rollouts = {};
            events.forEach(event => {
                if (!rollouts[event.rollout_id]) {
                    rollouts[event.rollout_id] = [];
                }
                rollouts[event.rollout_id].push(event);
            });
            
            // Draw timeline
            const height = canvas.height;
            const width = canvas.width;
            const rowHeight = 20;
            const padding = 10;
            
            ctx.clearRect(0, 0, width, height);
            
            let y = padding;
            Object.entries(rollouts).forEach(([rolloutId, rolloutEvents]) => {
                // Draw rollout label
                ctx.fillStyle = '#666';
                ctx.font = '10px monospace';
                ctx.fillText(rolloutId, 5, y + 12);
                
                // Draw events
                rolloutEvents.forEach((event, i) => {
                    if (i < rolloutEvents.length - 1) {
                        const nextEvent = rolloutEvents[i + 1];
                        const x1 = ((event.time - minTime) / duration) * (width - 100) + 90;
                        const x2 = ((nextEvent.time - minTime) / duration) * (width - 100) + 90;
                        
                        let color = '#666';
                        if (event.type.includes('generation')) color = '#4CAF50';
                        else if (event.type.includes('scoring')) color = '#2196F3';
                        
                        ctx.fillStyle = color;
                        ctx.fillRect(x1, y, x2 - x1, rowHeight - 2);
                    }
                });
                
                y += rowHeight;
            });
        }
        
        function drawRolloutsTable(rollouts) {
            let html = '<div class="rollout-row header">';
            html += '<div>ID</div><div>Prompt</div><div>Completion</div>';
            html += '<div>Reward</div><div>Gen Time</div><div>Score Time</div>';
            html += '</div>';
            
            rollouts.forEach(r => {
                const scores = r.scores ? JSON.parse(r.scores) : {};
                html += `<div class="rollout-row">`;
                html += `<div>${r.prompt_idx}:${r.generation_idx}</div>`;
                html += `<div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${r.prompt || ''}</div>`;
                html += `<div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${r.completion || ''}</div>`;
                html += `<div>${r.reward ? r.reward.toFixed(3) : '-'}</div>`;
                html += `<div>${r.generation_duration ? r.generation_duration.toFixed(2) + 's' : '-'}</div>`;
                html += `<div>${r.scoring_duration ? r.scoring_duration.toFixed(2) + 's' : '-'}</div>`;
                html += '</div>';
            });
            
            document.getElementById('rollouts-table').innerHTML = html;
        }
        
        function drawPerformanceStats(stats) {
            let html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">';
            
            // Generation stats
            html += '<div><h4>Generation Performance</h4>';
            if (stats.generation && stats.generation.count) {
                html += `<p>Count: ${stats.generation.count}</p>`;
                html += `<p>Avg Duration: ${stats.generation.avg_duration?.toFixed(2)}s</p>`;
                html += `<p>Min/Max: ${stats.generation.min_duration?.toFixed(2)}s / ${stats.generation.max_duration?.toFixed(2)}s</p>`;
                html += `<p>Avg Tokens/sec: ${stats.generation.avg_tps?.toFixed(1)}</p>`;
            } else {
                html += '<p>No data</p>';
            }
            html += '</div>';
            
            // Scoring stats
            html += '<div><h4>Scoring Performance</h4>';
            if (stats.scoring && stats.scoring.count) {
                html += `<p>Count: ${stats.scoring.count}</p>`;
                html += `<p>Avg Duration: ${stats.scoring.avg_duration?.toFixed(2)}s</p>`;
                html += `<p>Min/Max: ${stats.scoring.min_duration?.toFixed(2)}s / ${stats.scoring.max_duration?.toFixed(2)}s</p>`;
            } else {
                html += '<p>No data</p>';
            }
            html += '</div>';
            
            html += '</div>';
            document.getElementById('performance-stats').innerHTML = html;
        }
        
        // Make examples clickable to show batch details
        document.addEventListener('click', (e) => {
            const example = e.target.closest('.example');
            if (example && example.dataset.batchId) {
                selectBatch(parseInt(example.dataset.batchId));
            }
        });
    </script>
</body>
</html>
    '''


# API endpoints for telemetry data
@app.route('/api/batch/<int:batch_id>')
def get_batch_details(batch_id):
    """Get detailed information about a specific batch."""
    details = pipeline_tracker.db.get_batch_details(batch_id)
    if details:
        return jsonify(details)
    return jsonify({'error': 'Batch not found'}), 404

@app.route('/api/batches/recent')
def get_recent_batches():
    """Get recent batches with summary."""
    limit = request.args.get('limit', 10, type=int)
    batches = pipeline_tracker.db.get_recent_batches(limit)
    return jsonify({'batches': batches})

@app.route('/api/rollout/<rollout_id>')
def get_rollout_details(rollout_id):
    """Get detailed information about a specific rollout."""
    with pipeline_tracker.db._get_connection() as conn:
        rollout = conn.execute(
            "SELECT * FROM rollouts WHERE rollout_id = ?", (rollout_id,)
        ).fetchone()
        if rollout:
            return jsonify(dict(rollout))
    return jsonify({'error': 'Rollout not found'}), 404

@app.route('/api/batch/<int:batch_id>/timeline')
def get_batch_timeline(batch_id):
    """Get timeline of events for a batch."""
    timeline = pipeline_tracker.db.get_rollout_timeline(batch_id)
    return jsonify({'timeline': timeline})

@app.route('/api/performance')
def get_performance_stats():
    """Get performance statistics."""
    window = request.args.get('window_minutes', 5, type=int)
    stats = pipeline_tracker.db.get_performance_stats(window)
    return jsonify(stats)

@app.route('/api/metrics/training')
def get_training_metrics():
    """Get training metrics over time."""
    limit = request.args.get('limit', 100, type=int)
    with pipeline_tracker.db._get_connection() as conn:
        metrics = conn.execute("""
            SELECT * FROM training_metrics 
            ORDER BY step DESC 
            LIMIT ?
        """, (limit,)).fetchall()
    return jsonify({'metrics': [dict(m) for m in metrics]})

@app.route('/api/metrics/system')
def get_system_metrics():
    """Get system metrics over time."""
    minutes = request.args.get('minutes', 60, type=int)
    cutoff = time.time() - (minutes * 60)
    with pipeline_tracker.db._get_connection() as conn:
        metrics = conn.execute("""
            SELECT * FROM system_metrics 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, (cutoff,)).fetchall()
    return jsonify({'metrics': [dict(m) for m in metrics]})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('pipeline_update', pipeline_tracker.get_pipeline_state())


@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client."""
    emit('pipeline_update', pipeline_tracker.get_pipeline_state())


class GRPOWebDashboardCallback(TrainerCallback):
    """Callback that hooks into GRPO training to track the pipeline."""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.server_thread = None
        self.trainer = None
        self._batch_map = {}  # Map step to batch_id
        self.pipeline_tracker = pipeline_tracker  # Store reference to the global tracker
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Start the web server and setup tracking."""
        print(f"[DEBUG] GRPOWebDashboardCallback.on_train_begin called!")
        print(f"[DEBUG] Port: {self.port}")
        
        # Store trainer reference
        if hasattr(args, '_trainer'):
            self.trainer = args._trainer
            print(f"[DEBUG] Trainer reference stored")
        
        pipeline_tracker.total_steps = args.max_steps
        print(f"[DEBUG] Total steps: {args.max_steps}")
        
        # Start server
        print(f"[DEBUG] Starting web server thread...")
        self.server_thread = threading.Thread(
            target=lambda: socketio.run(app, host='0.0.0.0', port=self.port, debug=False, allow_unsafe_werkzeug=True),
            daemon=True
        )
        self.server_thread.start()
        
        time.sleep(2)  # Wait for server to start
        print(f"\n🌐 GRPO Pipeline Dashboard available at: http://localhost:{self.port}\n")
        print(f"[DEBUG] Server thread started: {self.server_thread.is_alive()}")
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Track batch creation and generation start."""
        # Batch creation and generation start is now handled by AsyncBatchGenerator
        pass
        
    def on_step_end(self, args, state, control, **kwargs):
        """Track training completion."""
        # Training completion is now handled by GRPOTrainer.compute_loss
        pass
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture metrics and batch transitions."""
        if logs:
            pipeline_tracker.update_metrics(state.global_step, logs)
            
            # Batch transitions are now handled by AsyncBatchGenerator and GRPOTrainer
                    
        # Emit update
        socketio.emit('pipeline_update', pipeline_tracker.get_pipeline_state(), namespace='/')


def create_grpo_web_dashboard(port: int = 5000) -> GRPOWebDashboardCallback:
    """Create a web dashboard callback for GRPO training."""
    return GRPOWebDashboardCallback(port=port)