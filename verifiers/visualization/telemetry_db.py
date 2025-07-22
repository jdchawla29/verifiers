"""SQLite database for GRPO training telemetry."""

import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RolloutInfo:
    """Information for a single rollout."""
    rollout_id: str  # batch_id:prompt_idx:generation_idx
    batch_id: int
    prompt_idx: int
    generation_idx: int
    prompt: str
    completion: str = ""
    
    # Timing information
    generation_start: Optional[float] = None
    generation_end: Optional[float] = None
    generation_duration: Optional[float] = None
    
    scoring_start: Optional[float] = None
    scoring_end: Optional[float] = None
    scoring_duration: Optional[float] = None
    
    # Scores and rewards
    reward: Optional[float] = None
    advantage: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Model information
    tokens_generated: Optional[int] = None
    tokens_per_second: Optional[float] = None
    
    # Training information
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    
    status: str = "pending"  # pending, generating, scoring, training, completed


class TelemetryDatabase:
    """SQLite database for storing detailed training telemetry."""
    
    def __init__(self, db_path: str = "grpo_telemetry.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Batches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id INTEGER PRIMARY KEY,
                    created_at REAL,
                    num_samples INTEGER,
                    num_generations INTEGER,
                    status TEXT,
                    
                    generation_start REAL,
                    generation_end REAL,
                    scoring_start REAL,
                    scoring_end REAL,
                    training_start REAL,
                    training_end REAL,
                    completed_at REAL,
                    
                    total_duration REAL,
                    generation_duration REAL,
                    scoring_duration REAL,
                    training_duration REAL
                )
            """)
            
            # Rollouts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollouts (
                    rollout_id TEXT PRIMARY KEY,
                    batch_id INTEGER,
                    prompt_idx INTEGER,
                    generation_idx INTEGER,
                    
                    prompt TEXT,
                    completion TEXT,
                    
                    generation_start REAL,
                    generation_end REAL,
                    generation_duration REAL,
                    
                    scoring_start REAL,
                    scoring_end REAL,
                    scoring_duration REAL,
                    
                    reward REAL,
                    advantage REAL,
                    scores TEXT,  -- JSON
                    
                    tokens_generated INTEGER,
                    tokens_per_second REAL,
                    
                    loss REAL,
                    grad_norm REAL,
                    
                    status TEXT,
                    
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                )
            """)
            
            # Training metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    step INTEGER PRIMARY KEY,
                    timestamp REAL,
                    batch_id INTEGER,
                    
                    loss REAL,
                    reward REAL,
                    reward_std REAL,
                    advantage REAL,
                    learning_rate REAL,
                    
                    kl_divergence REAL,
                    clip_ratio REAL,
                    
                    completions_per_second REAL,
                    tokens_per_second REAL,
                    
                    gpu_memory_used REAL,
                    gpu_utilization REAL,
                    
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                )
            """)
            
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp REAL PRIMARY KEY,
                    cpu_percent REAL,
                    memory_percent REAL,
                    gpu_memory_used REAL,
                    gpu_utilization REAL,
                    
                    active_batches INTEGER,
                    queued_batches INTEGER,
                    completed_batches INTEGER,
                    
                    avg_generation_time REAL,
                    avg_scoring_time REAL,
                    avg_training_time REAL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rollouts_batch ON rollouts(batch_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rollouts_status ON rollouts(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_step ON training_metrics(step)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with thread safety."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_batch(self, batch_id: int, num_samples: int, num_generations: int) -> None:
        """Create a new batch record."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO batches (
                        batch_id, created_at, num_samples, num_generations, status
                    ) VALUES (?, ?, ?, ?, ?)
                """, (batch_id, time.time(), num_samples, num_generations, 'queued'))
                conn.commit()
    
    def create_rollout(self, rollout_info: RolloutInfo) -> None:
        """Create a new rollout record."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO rollouts (
                        rollout_id, batch_id, prompt_idx, generation_idx,
                        prompt, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rollout_info.rollout_id,
                    rollout_info.batch_id,
                    rollout_info.prompt_idx,
                    rollout_info.generation_idx,
                    rollout_info.prompt,
                    rollout_info.status
                ))
                conn.commit()
    
    def update_rollout_generation(self, rollout_id: str, completion: str, 
                                start_time: float, end_time: float,
                                tokens_generated: Optional[int] = None) -> None:
        """Update rollout with generation results."""
        duration = end_time - start_time
        tokens_per_second = tokens_generated / duration if tokens_generated and duration > 0 else None
        
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE rollouts SET
                        completion = ?,
                        generation_start = ?,
                        generation_end = ?,
                        generation_duration = ?,
                        tokens_generated = ?,
                        tokens_per_second = ?,
                        status = 'generated'
                    WHERE rollout_id = ?
                """, (
                    completion, start_time, end_time, duration,
                    tokens_generated, tokens_per_second, rollout_id
                ))
                conn.commit()
    
    def update_rollout_scoring(self, rollout_id: str, reward: float,
                             advantage: float, scores: Dict[str, float],
                             start_time: float, end_time: float) -> None:
        """Update rollout with scoring results."""
        duration = end_time - start_time
        
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE rollouts SET
                        reward = ?,
                        advantage = ?,
                        scores = ?,
                        scoring_start = ?,
                        scoring_end = ?,
                        scoring_duration = ?,
                        status = 'scored'
                    WHERE rollout_id = ?
                """, (
                    reward, advantage, json.dumps(scores),
                    start_time, end_time, duration, rollout_id
                ))
                conn.commit()
    
    def update_rollout_training(self, rollout_id: str, loss: float,
                              grad_norm: Optional[float] = None) -> None:
        """Update rollout with training results."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE rollouts SET
                        loss = ?,
                        grad_norm = ?,
                        status = 'completed'
                    WHERE rollout_id = ?
                """, (loss, grad_norm, rollout_id))
                conn.commit()
    
    def update_batch_status(self, batch_id: int, status: str,
                          **timing_kwargs) -> None:
        """Update batch status and timing information."""
        with self._lock:
            with self._get_connection() as conn:
                # Build update query dynamically based on provided timings
                updates = ["status = ?"]
                values = [status]
                
                for key, value in timing_kwargs.items():
                    if key in ['generation_start', 'generation_end', 'scoring_start',
                             'scoring_end', 'training_start', 'training_end', 'completed_at']:
                        updates.append(f"{key} = ?")
                        values.append(value)
                
                # Calculate durations if we have start/end times
                if 'generation_start' in timing_kwargs and 'generation_end' in timing_kwargs:
                    duration = timing_kwargs['generation_end'] - timing_kwargs['generation_start']
                    updates.append("generation_duration = ?")
                    values.append(duration)
                
                if 'scoring_start' in timing_kwargs and 'scoring_end' in timing_kwargs:
                    duration = timing_kwargs['scoring_end'] - timing_kwargs['scoring_start']
                    updates.append("scoring_duration = ?")
                    values.append(duration)
                
                if 'training_start' in timing_kwargs and 'training_end' in timing_kwargs:
                    duration = timing_kwargs['training_end'] - timing_kwargs['training_start']
                    updates.append("training_duration = ?")
                    values.append(duration)
                
                values.append(batch_id)
                
                query = f"UPDATE batches SET {', '.join(updates)} WHERE batch_id = ?"
                conn.execute(query, values)
                conn.commit()
    
    def log_training_metrics(self, step: int, batch_id: int, metrics: Dict[str, float]) -> None:
        """Log training metrics for a step."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO training_metrics (
                        step, timestamp, batch_id, loss, reward, reward_std,
                        advantage, learning_rate, kl_divergence, clip_ratio,
                        completions_per_second, tokens_per_second,
                        gpu_memory_used, gpu_utilization
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    step, time.time(), batch_id,
                    metrics.get('loss'),
                    metrics.get('reward'),
                    metrics.get('reward_std'),
                    metrics.get('advantage'),
                    metrics.get('learning_rate'),
                    metrics.get('kl_divergence'),
                    metrics.get('clip_ratio'),
                    metrics.get('completions_per_second'),
                    metrics.get('tokens_per_second'),
                    metrics.get('gpu_memory_used'),
                    metrics.get('gpu_utilization')
                ))
                conn.commit()
    
    def log_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Log system-wide metrics."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent,
                        gpu_memory_used, gpu_utilization,
                        active_batches, queued_batches, completed_batches,
                        avg_generation_time, avg_scoring_time, avg_training_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    metrics.get('cpu_percent'),
                    metrics.get('memory_percent'),
                    metrics.get('gpu_memory_used'),
                    metrics.get('gpu_utilization'),
                    metrics.get('active_batches'),
                    metrics.get('queued_batches'),
                    metrics.get('completed_batches'),
                    metrics.get('avg_generation_time'),
                    metrics.get('avg_scoring_time'),
                    metrics.get('avg_training_time')
                ))
                conn.commit()
    
    def get_batch_details(self, batch_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a batch."""
        with self._get_connection() as conn:
            # Get batch info
            batch = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            
            if not batch:
                return None
            
            # Get rollouts
            rollouts = conn.execute(
                "SELECT * FROM rollouts WHERE batch_id = ? ORDER BY prompt_idx, generation_idx",
                (batch_id,)
            ).fetchall()
            
            return {
                'batch': dict(batch),
                'rollouts': [dict(r) for r in rollouts]
            }
    
    def get_recent_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent batches with summary information."""
        with self._get_connection() as conn:
            batches = conn.execute("""
                SELECT b.*, 
                       COUNT(r.rollout_id) as rollout_count,
                       AVG(r.reward) as avg_reward,
                       AVG(r.generation_duration) as avg_gen_time,
                       AVG(r.scoring_duration) as avg_score_time
                FROM batches b
                LEFT JOIN rollouts r ON b.batch_id = r.batch_id
                GROUP BY b.batch_id
                ORDER BY b.created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [dict(b) for b in batches]
    
    def get_rollout_timeline(self, batch_id: int) -> List[Dict[str, Any]]:
        """Get timeline of events for all rollouts in a batch."""
        with self._get_connection() as conn:
            events = []
            
            # Get all timing events
            rollouts = conn.execute("""
                SELECT rollout_id, prompt_idx, generation_idx,
                       generation_start, generation_end,
                       scoring_start, scoring_end
                FROM rollouts
                WHERE batch_id = ?
            """, (batch_id,)).fetchall()
            
            for r in rollouts:
                r_dict = dict(r)
                rollout_id = r_dict['rollout_id']
                
                # Add generation events
                if r_dict['generation_start']:
                    events.append({
                        'time': r_dict['generation_start'],
                        'type': 'generation_start',
                        'rollout_id': rollout_id,
                        'prompt_idx': r_dict['prompt_idx'],
                        'generation_idx': r_dict['generation_idx']
                    })
                
                if r_dict['generation_end']:
                    events.append({
                        'time': r_dict['generation_end'],
                        'type': 'generation_end',
                        'rollout_id': rollout_id,
                        'prompt_idx': r_dict['prompt_idx'],
                        'generation_idx': r_dict['generation_idx']
                    })
                
                # Add scoring events
                if r_dict['scoring_start']:
                    events.append({
                        'time': r_dict['scoring_start'],
                        'type': 'scoring_start',
                        'rollout_id': rollout_id,
                        'prompt_idx': r_dict['prompt_idx'],
                        'generation_idx': r_dict['generation_idx']
                    })
                
                if r_dict['scoring_end']:
                    events.append({
                        'time': r_dict['scoring_end'],
                        'type': 'scoring_end',
                        'rollout_id': rollout_id,
                        'prompt_idx': r_dict['prompt_idx'],
                        'generation_idx': r_dict['generation_idx']
                    })
            
            # Sort by time
            events.sort(key=lambda x: x['time'])
            
            return events
    
    def get_performance_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance statistics for the recent time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._get_connection() as conn:
            # Get generation stats
            gen_stats = conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    AVG(generation_duration) as avg_duration,
                    MIN(generation_duration) as min_duration,
                    MAX(generation_duration) as max_duration,
                    AVG(tokens_per_second) as avg_tps
                FROM rollouts
                WHERE generation_end > ? AND generation_duration IS NOT NULL
            """, (cutoff_time,)).fetchone()
            
            # Get scoring stats
            score_stats = conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    AVG(scoring_duration) as avg_duration,
                    MIN(scoring_duration) as min_duration,
                    MAX(scoring_duration) as max_duration
                FROM rollouts
                WHERE scoring_end > ? AND scoring_duration IS NOT NULL
            """, (cutoff_time,)).fetchone()
            
            # Get batch stats
            batch_stats = conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    AVG(total_duration) as avg_duration,
                    AVG(generation_duration) as avg_gen_duration,
                    AVG(scoring_duration) as avg_score_duration,
                    AVG(training_duration) as avg_train_duration
                FROM batches
                WHERE completed_at > ? AND total_duration IS NOT NULL
            """, (cutoff_time,)).fetchone()
            
            return {
                'generation': dict(gen_stats) if gen_stats else {},
                'scoring': dict(score_stats) if score_stats else {},
                'batches': dict(batch_stats) if batch_stats else {}
            }
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """Remove old telemetry data."""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        with self._lock:
            with self._get_connection() as conn:
                # Delete old batches and cascading rollouts
                conn.execute(
                    "DELETE FROM batches WHERE created_at < ?", (cutoff_time,)
                )
                
                # Delete old system metrics
                conn.execute(
                    "DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,)
                )
                
                # Delete old training metrics
                conn.execute(
                    "DELETE FROM training_metrics WHERE timestamp < ?", (cutoff_time,)
                )
                
                conn.commit()
                
                # Vacuum to reclaim space
                conn.execute("VACUUM")