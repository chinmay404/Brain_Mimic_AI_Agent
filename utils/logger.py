import os
import json
import logging
import datetime
import uuid
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass

class ResearchLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ResearchLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = None):
        if hasattr(self, 'initialized'):
            return
        
        # Determine log directory
        if log_dir is None:
            # Default to brain_working/logs relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir) # brain_working
            self.log_dir = os.path.join(project_root, "logs")
        else:
            self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Generate session ID and filenames
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.log_file_json = os.path.join(self.log_dir, f"session_{self.session_id}.jsonl")
        self.log_file_txt = os.path.join(self.log_dir, f"session_{self.session_id}.log")

        # Setup Python logging for console and text file
        self.logger = logging.getLogger("BrainLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = [] # Clear existing handlers

        # File Handler (Text)
        file_handler = logging.FileHandler(self.log_file_txt)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s [%(name)s]: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.initialized = True
        self.log_system_event("LoggerInitialized", {"log_dir": self.log_dir, "session_id": self.session_id})
        print(f"📝 Research Logger initialized. Logs at: {self.log_dir}")

    def _serialize(self, obj):
        """Helper to serialize objects for JSON."""
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def log(self, component: str, event_type: str, message: str, data: Optional[Dict[str, Any]] = None, level: str = "INFO"):
        """
        General logging method.
        
        Args:
            component: The part of the system (e.g., 'DLPFC', 'OFC', 'Thalamus').
            event_type: Category of event (e.g., 'Decision', 'Input', 'Error').
            message: Human readable message.
            data: Structured data for research analysis.
            level: Log level (INFO, WARNING, ERROR, DEBUG).
        """
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "component": component,
            "event_type": event_type,
            "message": message,
            "level": level,
            "data": data or {}
        }

        # Write to JSONL (Structured Log)
        try:
            with open(self.log_file_json, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=self._serialize) + "\n")
        except Exception as e:
            print(f"Error writing to JSON log: {e}")

        # Write to Text Log / Console
        log_msg = f"[{component}] {event_type}: {message}"
        if level == "INFO":
            self.logger.info(log_msg)
        elif level == "WARNING":
            self.logger.warning(log_msg)
        elif level == "ERROR":
            self.logger.error(log_msg)
        elif level == "DEBUG":
            self.logger.debug(log_msg)

    def log_execution(self, component: str, input_data: Any, output_data: Any, metadata: Dict[str, Any] = None):
        """Specialized method for logging execution steps (Input -> Output)."""
        self.log(
            component=component,
            event_type="Execution",
            message="Executed processing step",
            data={
                "input": input_data,
                "output": output_data,
                "metadata": metadata or {}
            }
        )

    def log_system_event(self, event_name: str, details: Dict[str, Any] = None):
        """Log system level events."""
        self.log("System", "SystemEvent", event_name, details)

# Global instance accessor
def get_logger():
    return ResearchLogger()
