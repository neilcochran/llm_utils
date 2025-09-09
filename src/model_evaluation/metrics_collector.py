"""Metrics collection for model evaluation - data only."""

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class TokenMetrics:
    """Token-level statistics for model output."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    average_token_length: float
    tokens_per_second: float
    peak_tokens_per_second: float
    tokens_per_second_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class TimingMetrics:
    """Timing measurements for model inference."""
    time_to_first_token_ms: float
    time_to_completion_ms: float
    total_inference_time_ms: float
    prompt_processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCollector:
    """Collects and calculates performance metrics during inference."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.tokens_received: List[str] = []
        self.token_timestamps: List[float] = []
        self.prompt_tokens: int = 0
    
    def start_inference(self) -> None:
        """Mark the start of inference."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = None
        self.tokens_received = []
        self.token_timestamps = []
    
    def record_first_token(self) -> None:
        """Mark when first token is received."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
    
    def record_token(self, token: str) -> None:
        """Record a received token."""
        current_time = time.perf_counter()
        if self.first_token_time is None:
            self.record_first_token()
        self.tokens_received.append(token)
        self.token_timestamps.append(current_time)
    
    def end_inference(self) -> None:
        """Mark the end of inference."""
        self.end_time = time.perf_counter()
    
    def set_prompt_tokens(self, count: int) -> None:
        """Set the number of prompt tokens."""
        self.prompt_tokens = count
    
    def calculate_timing_metrics(self) -> TimingMetrics:
        """Calculate timing metrics from recorded data."""
        if not self.start_time or not self.end_time:
            raise ValueError("Inference timing not properly recorded")
        
        total_time_ms = (self.end_time - self.start_time) * 1000
        
        if self.first_token_time:
            time_to_first_token_ms = (self.first_token_time - self.start_time) * 1000
            time_to_completion_ms = (self.end_time - self.first_token_time) * 1000
        else:
            time_to_first_token_ms = total_time_ms
            time_to_completion_ms = 0.0
        
        return TimingMetrics(
            time_to_first_token_ms=time_to_first_token_ms,
            time_to_completion_ms=time_to_completion_ms,
            total_inference_time_ms=total_time_ms
        )
    
    def calculate_token_metrics(self) -> TokenMetrics:
        """Calculate token metrics from recorded data."""
        completion_tokens = len(self.tokens_received)
        total_tokens = self.prompt_tokens + completion_tokens
        
        if completion_tokens == 0:
            return TokenMetrics(
                total_tokens=total_tokens,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=0,
                average_token_length=0.0,
                tokens_per_second=0.0,
                peak_tokens_per_second=0.0,
                tokens_per_second_variance=0.0
            )
        
        # Calculate average token length
        total_chars = sum(len(token) for token in self.tokens_received)
        avg_token_length = total_chars / completion_tokens if completion_tokens > 0 else 0.0
        
        # Calculate average tokens per second
        if self.start_time and self.end_time:
            total_time_seconds = self.end_time - self.start_time
            tokens_per_second = completion_tokens / total_time_seconds if total_time_seconds > 0 else 0.0
        else:
            tokens_per_second = 0.0
        
        # Calculate peak tokens/second and variance using sliding window
        peak_tokens_per_second = 0.0
        tokens_per_second_variance = 0.0
        
        if len(self.token_timestamps) > 1:
            # Calculate instantaneous throughput using 1-second sliding windows
            window_size = 1.0  # 1 second window
            throughput_samples = []
            
            for i in range(len(self.token_timestamps)):
                current_time = self.token_timestamps[i]
                # Count tokens in the 1-second window ending at current_time
                tokens_in_window = 0
                for j in range(len(self.token_timestamps)):
                    if current_time - window_size <= self.token_timestamps[j] <= current_time:
                        tokens_in_window += 1
                
                if tokens_in_window > 0:
                    # Calculate throughput for this window
                    window_throughput = tokens_in_window / window_size
                    throughput_samples.append(window_throughput)
                    peak_tokens_per_second = max(peak_tokens_per_second, window_throughput)
            
            # Calculate variance of throughput samples
            if len(throughput_samples) > 1:
                mean_throughput = sum(throughput_samples) / len(throughput_samples)
                variance = sum((x - mean_throughput) ** 2 for x in throughput_samples) / len(throughput_samples)
                tokens_per_second_variance = variance ** 0.5  # Standard deviation
        
        return TokenMetrics(
            total_tokens=total_tokens,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_tokens,
            average_token_length=avg_token_length,
            tokens_per_second=tokens_per_second,
            peak_tokens_per_second=peak_tokens_per_second,
            tokens_per_second_variance=tokens_per_second_variance
        )