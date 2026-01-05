import time
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional


logger = logging.getLogger("TickEngine")


@dataclass
class TickData:
    tick_id: int
    timestamp: float
    delta_time: float
    inputs: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)


class InputSource(ABC):
    """Interface for anything that provides data (Sensors, API, User)."""
    @abstractmethod
    def fetch_inputs(self, tick_id: int) -> Dict[str, Any]:
        pass


class BrainCore(ABC):
    """Interface for the logic unit (The Agent)."""
    @abstractmethod
    def process_tick(self, data: TickData) -> List[str]:
        """Takes current state, returns list of actions."""
        pass


class ActionHandler(ABC):
    """Interface for executing results (Robot arm, API call, Printer)."""
    @abstractmethod
    def execute(self, actions: List[str], tick_id: int):
        pass


class UniversalTickEngine:
    def __init__(self,
                 brain: BrainCore,
                 inputs: List[InputSource],
                 outputs: List[ActionHandler],
                 tick_rate_hz: float = 1.0) -> None:
        self.brain = brain
        self.inputs = inputs
        self.outputs = outputs
        self.target_interval = 1.0 / tick_rate_hz
        self.tick_count = 0
        self.running = False
        self.last_time = time.time()

    
    
    def run(self, max_ticks: Optional[int] = None):
        self.running = True
        logger.info(f"Engine Started : rate : {1/self.target_interval} Hz")
        
        while self.running:
            if max_ticks is not None and self.tick_count >= max_ticks:
                logger.info("Max ticks reached. Stopping engine.")
                self.running = False
                break

            start_time = time.time()
            delta_time = start_time - self.last_time
            self.last_time = start_time
            try:
                tick_data = TickData(
                    tick_id=self.tick_count,
                    timestamp=start_time,
                    delta_time=delta_time
                )       
            except Exception as e:
                logger.error(f"[Engine Error] Tick Data Formation Error : {e}")
                
            try:
                for source in self.inputs:
                    new_data = source.fetch_inputs(self.tick_count)
                    if new_data:
                        tick_data.inputs.update(new_data)
                actions = self.brain.process_tick(tick_data)
                tick_data.actions = actions
                
                if actions:
                    for handler in self.outputs:
                        handler.execute(actions, self.tick_count)
                    
            except Exception as e :
                logger.error(f"[Engine Error] CRITICAL ERROR ON TICK {self.tick_count}: {e}")
            
            
            self.tick_count += 1
            processing_time = time.time() - start_time
            sleep_needed = self.target_interval - processing_time
            
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            else:
                logger.warning(f"Tick {self.tick_count} lagging by {abs(sleep_needed):.4f}s")