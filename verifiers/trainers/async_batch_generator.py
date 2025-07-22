import asyncio
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from verifiers import GenerateOutputs


@dataclass
class BatchRequest:
    """Request for batch generation"""

    batch_id: int
    env_inputs: Dict[str, List[Any] | None]
    processing_class: Any
    mask_env_responses: bool
    max_seq_len: int
    mask_truncated_completions: bool
    zero_truncated_completions: bool
    max_concurrent: int


@dataclass
class BatchResult:
    """Result from batch generation"""

    batch_id: int
    processed_results: Dict[str, Any]
    generation_time: float = 0.0
    all_reward_dict: Dict[str, List[float]] = field(
        default_factory=dict
    )  # All reward scores
    completions: List[Any] = field(
        default_factory=list
    )  # Store completions for logging
    prompts: List[Any] = field(default_factory=list)  # Store prompts for logging


class AsyncBatchGenerator:
    """
    Manages asynchronous batch generation for GRPO training.

    This class runs generation in a separate thread, allowing training to continue
    while future batches are being generated. It maintains a queue of pending
    generation requests and completed results.
    """

    def __init__(
        self,
        env,
        client_config,
        model_name: str,
        sampling_args: Dict[str, Any],
        num_batches_ahead: int = 1,
        max_queue_size: Optional[int] = None,
        generation_timeout: float = 300.0,  # 5 minutes default
        pipeline_tracker: Optional[Any] = None,  # Optional tracker for dashboard
    ):
        self.env = env
        self.client_config = client_config
        self.client = None  # Will be created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.num_batches_ahead = num_batches_ahead
        self.max_queue_size = max_queue_size or max(num_batches_ahead * 2, 4)
        self.generation_timeout = generation_timeout
        self.pipeline_tracker = pipeline_tracker  # Store tracker reference

        # Queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False

        # Tracking
        self.pending_batches = set()  # batch_ids currently being processed
        self.completed_batches = {}  # batch_id -> BatchResult
        self.next_expected_batch = 0
        self.generation_times = deque(maxlen=100)  # Track recent generation times

        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(f"AsyncBatchGenerator-{id(self)}")
        self.is_generating = False  # Track if currently generating
        self.worker_loop = None  # Will be set in worker thread
        self.started = False  # Track if generator is started

        # Synchronization
        self._lock = threading.Lock()
        
        # Batch ID mapping for tracker
        self._batch_id_map = {}

    def start(self):
        """Start the async generation worker thread"""
        if self.started:
            return

        self.worker_thread = threading.Thread(
            target=self._generation_worker, daemon=True, name="AsyncBatchGenerator"
        )
        self.worker_thread.start()
        self.started = True

    def stop(self):
        """Stop the async generation worker thread"""
        if not self.started:
            return

        self.stop_event.set()
        # Send poison pill
        self.request_queue.put(None)

        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

        self.started = False

    def submit_batch(self, request: BatchRequest) -> bool:
        """
        Submit a batch for async generation.

        Returns:
            bool: True if submitted successfully, False if queue is full
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        with self._lock:
            if request.batch_id in self.pending_batches:
                return True  # Already submitted

            if len(self.pending_batches) >= self.max_queue_size:
                return False  # Queue full

            self.pending_batches.add(request.batch_id)

        self.request_queue.put(request)
        
        # Notify tracker that batch was queued
        if self.pipeline_tracker:
            # Get number of samples from the inputs
            # Try 'prompt' first (used by GRPO), then 'text' as fallback
            prompts = request.env_inputs.get('prompt', request.env_inputs.get('text', []))
            num_samples = len(prompts) if prompts else 0
            # Assume 1 generation per prompt for now
            batch_info_id = self.pipeline_tracker.create_batch(num_samples, 1)
            # Map our batch_id to tracker's batch_id (thread-safe)
            with self._lock:
                self._batch_id_map[request.batch_id] = batch_info_id
        
        return True

    def get_batch(self, batch_id: int, timeout: Optional[float] = None) -> BatchResult:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait (uses generation_timeout if None)

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If generation failed
        """
        timeout = timeout or self.generation_timeout
        start_time = time.time()

        while True:
            # Check if already completed
            with self._lock:
                if batch_id in self.completed_batches:
                    return self.completed_batches.pop(batch_id)

            # Check for new results
            try:
                result = self.result_queue.get(timeout=0.1)
                with self._lock:
                    self.completed_batches[result.batch_id] = result
                    self.pending_batches.discard(result.batch_id)

                # If this is our batch, return it
                if result.batch_id == batch_id:
                    with self._lock:
                        return self.completed_batches.pop(batch_id)

            except queue.Empty:
                pass

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} generation timed out after {timeout}s"
                )

    def get_tracker_batch_id(self, batch_id: int) -> Optional[int]:
        """Get the tracker's batch ID for a given batch ID."""
        with self._lock:
            return self._batch_id_map.get(batch_id)
    
    def get_pending_count(self) -> int:
        """Get number of batches currently being generated"""
        with self._lock:
            return len(self.pending_batches)

    def get_completed_count(self) -> int:
        """Get number of completed batches waiting to be retrieved"""
        with self._lock:
            return len(self.completed_batches)

    def get_average_generation_time(self) -> float:
        """Get average generation time for recent batches"""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    def should_submit_more(self) -> bool:
        """Check if we should submit more batches for generation"""
        with self._lock:
            total_pending = len(self.pending_batches) + len(self.completed_batches)
            return total_pending < self.num_batches_ahead

    def _generation_worker(self):
        """Worker thread that processes generation requests"""
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop  # Store the event loop reference

        # Create the AsyncOpenAI client within this event loop
        import httpx
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=self.client_config["base_url"],
            api_key=self.client_config["api_key"],
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.client_config["http_client_args"]["limits"][
                        "max_connections"
                    ]
                ),
                timeout=self.client_config["http_client_args"]["timeout"],
            ),
        )

        try:
            while not self.stop_event.is_set():
                try:
                    # Get next request
                    request = self.request_queue.get(timeout=0.1)
                    if request is None:  # Poison pill
                        break

                    # Generate batch using the async method
                    start_time = time.time()
                    result = loop.run_until_complete(
                        self._generate_batch_async(request)
                    )
                    generation_time = time.time() - start_time
                    result.generation_time = generation_time
                    self.generation_times.append(generation_time)
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            # Clean up the client
            if self.client:
                loop.run_until_complete(self.client.close())
            # Clean up the event loop
            loop.close()
            asyncio.set_event_loop(None)

    async def _generate_batch_async(self, request: BatchRequest) -> BatchResult:
        """
        Generate a single batch asynchronously.
        """
        # Notify tracker that generation is starting
        if self.pipeline_tracker:
            with self._lock:
                tracker_batch_id = self._batch_id_map.get(request.batch_id)
            if tracker_batch_id is not None:
                self.pipeline_tracker.start_generation(tracker_batch_id)
        
        # Call environment generation
        self.is_generating = True
        generation_start_time = time.time()
        env_results = await self.env.a_generate(
            request.env_inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=request.max_concurrent,
        )
        generation_end_time = time.time()
        self.is_generating = False
        
        # Notify tracker that generation is complete
        if self.pipeline_tracker:
            with self._lock:
                tracker_batch_id = self._batch_id_map.get(request.batch_id)
            if tracker_batch_id is not None:
                prompts = env_results["prompt"]
                completions = env_results["completion"]
                
                # Extract readable prompts for display
                display_prompts = []
                for prompt in prompts:
                    if isinstance(prompt, list):
                        # Handle message format prompts (like DocVQA)
                        for msg in prompt:
                            if msg.get('role') == 'user':
                                content = msg.get('content', [])
                                if isinstance(content, list):
                                    # Extract text from content blocks
                                    for item in content:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            display_prompts.append(item.get('text', ''))
                                            break
                                elif isinstance(content, str):
                                    display_prompts.append(content)
                                break
                    elif isinstance(prompt, str):
                        display_prompts.append(prompt)
                    else:
                        display_prompts.append(str(prompt))
                
                # If no prompts extracted, use placeholders
                if not display_prompts:
                    display_prompts = [f"Prompt {i+1}" for i in range(len(prompts))]
                
                # Extract readable completions for display
                display_completions = []
                
                # Check if we have unique prompts or if they're repeated
                unique_prompts = []
                seen = set()
                for i, p in enumerate(display_prompts):
                    if p not in seen:
                        unique_prompts.append((i, p))
                        seen.add(p)
                
                # If prompts are repeated (one per generation), group completions accordingly
                if len(unique_prompts) < len(display_prompts):
                    # Prompts are repeated - group completions by unique prompt
                    num_unique = len(unique_prompts)
                    num_generations = len(display_prompts) // num_unique if num_unique > 0 else 1
                    
                    for i, (orig_idx, prompt) in enumerate(unique_prompts):
                        prompt_completions = []
                        for j in range(num_generations):
                            idx = i * num_generations + j
                            if idx < len(completions):
                                completion = completions[idx]
                                if isinstance(completion, list):
                                    # Handle message format completions
                                    for msg in completion:
                                        if msg.get('role') == 'assistant':
                                            prompt_completions.append(msg.get('content', ''))
                                            break
                                    else:
                                        prompt_completions.append('')
                                elif isinstance(completion, str):
                                    prompt_completions.append(completion)
                                else:
                                    prompt_completions.append(str(completion))
                        display_completions.append(prompt_completions)
                    
                    # Update display_prompts to only show unique ones
                    display_prompts = [p for _, p in unique_prompts]
                else:
                    # Each prompt is unique - assume one completion per prompt
                    for completion in completions:
                        if isinstance(completion, list):
                            # Handle message format completions
                            comp_text = ''
                            for msg in completion:
                                if msg.get('role') == 'assistant':
                                    comp_text = msg.get('content', '')
                                    break
                            display_completions.append([comp_text])
                        elif isinstance(completion, str):
                            display_completions.append([completion])
                        else:
                            display_completions.append([str(completion)])
                
                # Pass generation timing info
                self.pipeline_tracker.complete_generation_with_timing(
                    tracker_batch_id, display_prompts, display_completions,
                    generation_start_time, generation_end_time
                )

        # Notify tracker that scoring is starting
        if self.pipeline_tracker:
            with self._lock:
                tracker_batch_id = self._batch_id_map.get(request.batch_id)
            if tracker_batch_id is not None:
                self.pipeline_tracker.start_scoring(tracker_batch_id)
        
        # Extract all reward-related keys
        all_reward_dict = {}
        reward_keys = [
            k for k in env_results.keys() if k.endswith("_func") or k == "reward"
        ]
        for key in reward_keys:
            all_reward_dict[key] = env_results[key]

        # Process results
        processed_results = self.env.process_env_results(
            env_results["prompt"],
            env_results.get("images"),  # May be None for text-only tasks
            env_results["completion"],
            env_results["state"],
            env_results["reward"],
            processing_class=request.processing_class,
            max_seq_len=request.max_seq_len,
            mask_env_responses=request.mask_env_responses,
            mask_truncated_completions=request.mask_truncated_completions,
            zero_truncated_completions=request.zero_truncated_completions,
        )
        
        # Notify tracker that scoring is complete
        if self.pipeline_tracker:
            with self._lock:
                tracker_batch_id = self._batch_id_map.get(request.batch_id)
            if tracker_batch_id is not None:
                rewards = env_results["reward"]
                # Calculate advantages if available
                advantages = processed_results.get("advantages", [0] * len(rewards))
                # Get all scores from reward dict
                scores = []
                for i in range(len(rewards)):
                    example_scores = []
                    for key in reward_keys:
                        if key in all_reward_dict and i < len(all_reward_dict[key]):
                            example_scores.append(all_reward_dict[key][i])
                    scores.append(example_scores)
                
                self.pipeline_tracker.complete_scoring(tracker_batch_id, rewards, advantages, scores)

        return BatchResult(
            batch_id=request.batch_id,
            processed_results=processed_results,
            all_reward_dict=all_reward_dict,
            completions=env_results["completion"],
            prompts=env_results["prompt"],
        )

    async def _evaluate_async(self, num_samples: int = -1) -> GenerateOutputs:
        """
        Run evaluation in the worker thread's event loop.
        """
        # Get evaluation dataset
        if self.env.eval_dataset is None:
            self.env.logger.info(
                "eval_dataset is not set, falling back to train dataset"
            )
            assert self.env.dataset is not None
            inputs = self.env.get_dataset(n=num_samples)
        else:
            inputs = self.env.get_eval_dataset(n=num_samples)
        assert inputs is not None, "No dataset found"

        # Run generation on eval dataset
        results = await self.env.a_generate(
            inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
        )
        return results

    def evaluate(self, num_samples: int = -1) -> GenerateOutputs:
        """
        Run evaluation synchronously by creating a separate thread with its own event loop.
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        # Run evaluation in a separate thread to avoid event loop conflicts
        result_container = []
        exception_container = []

        def run_evaluation():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create a new client for this evaluation
            import httpx
            from openai import AsyncOpenAI

            eval_client = AsyncOpenAI(
                base_url=self.client_config["base_url"],
                api_key=self.client_config["api_key"],
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.client_config["http_client_args"][
                            "limits"
                        ]["max_connections"]
                    ),
                    timeout=self.client_config["http_client_args"]["timeout"],
                ),
            )

            async def run_eval():
                try:
                    # Get evaluation dataset
                    if self.env.eval_dataset is None:
                        self.env.logger.info(
                            "eval_dataset is not set, falling back to train dataset"
                        )
                        assert self.env.dataset is not None
                        inputs = self.env.get_dataset(n=num_samples)
                    else:
                        inputs = self.env.get_eval_dataset(n=num_samples)
                    assert inputs is not None, "No dataset found"

                    # Run generation on eval dataset
                    results = await self.env.a_generate(
                        inputs,
                        client=eval_client,
                        model=self.model_name,
                        sampling_args=self.sampling_args,
                    )
                    result_container.append(results)
                except Exception as e:
                    exception_container.append(e)
                finally:
                    await eval_client.close()

            try:
                loop.run_until_complete(run_eval())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        # Run evaluation in a separate thread
        import threading

        eval_thread = threading.Thread(target=run_evaluation)
        eval_thread.start()
        eval_thread.join(timeout=self.generation_timeout)

        if eval_thread.is_alive():
            raise TimeoutError(f"Evaluation timed out after {self.generation_timeout}s")

        if exception_container:
            raise exception_container[0]

        if not result_container:
            raise RuntimeError("Evaluation completed but no results were returned")

        return result_container[0]
