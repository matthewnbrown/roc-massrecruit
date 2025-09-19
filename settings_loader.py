#!/usr/bin/env python3
"""
Settings loader utility for reading configuration from settings.json
"""

import json
import os
from typing import Dict, Any, Optional


class SettingsLoader:
    """Load and manage settings from JSON configuration file"""
    
    def __init__(self, settings_file: str = "settings.json"):
        """
        Initialize settings loader
        
        Args:
            settings_file: Path to the settings JSON file
        """
        self.settings_file = settings_file
        self._settings = None
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from JSON file"""
        try:
            if not os.path.exists(self.settings_file):
                raise FileNotFoundError(f"Settings file not found: {self.settings_file}")
            
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                self._settings = json.load(f)
                
            print(f"✅ Loaded settings from {self.settings_file}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in settings file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load settings: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation (e.g., 'database.path')
        
        Args:
            key_path: Dot-separated path to the setting
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        if self._settings is None:
            return default
        
        keys = key_path.split('.')
        value = self._settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_database_path(self) -> str:
        """Get database path"""
        return self.get('database.path', 'accounts.db')
    
    def get_base_url(self) -> str:
        """Get base URL"""
        return self.get('server.base_url', "localhost:8002")
    
    def get_recruit_url(self) -> str:
        """Get recruit URL"""
        base_url = self.get_base_url()
        recruit_endpoint = self.get('server.recruit_endpoint', 'recruiter.php')
        return f"{base_url.rstrip('/')}/{recruit_endpoint}"
    
    def get_login_url(self) -> str:
        """Get login URL"""
        base_url = self.get_base_url()
        login_endpoint = self.get('server.login_endpoint', 'login.php')
        return f"{base_url.rstrip('/')}/{login_endpoint}"
    
    def get_model_path(self) -> str:
        """Get model path"""
        return self.get('model.path', 'models/2025_09_12best_model.pth')
    
    def get_model_device(self) -> str:
        """Get model device"""
        return self.get('model.device', 'cuda')
    
    def get_captcha_api_url(self) -> str:
        """Get CAPTCHA API URL"""
        return self.get('captcha.api_url', 'http://localhost:5000/solve')
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold"""
        return self.get('captcha.confidence_threshold', 0.8)
    
    def get_max_attempts(self) -> int:
        """Get max CAPTCHA attempts"""
        return self.get('captcha.max_attempts', 2)
    
    def get_use_captcha(self) -> bool:
        """Get whether to use CAPTCHA"""
        return self.get('captcha.use_captcha', False)
    
    def get_captcha_messages(self) -> Dict[str, str]:
        """Get CAPTCHA-related messages"""
        return {
            'unsolved': self.get('captcha.unsolved_message', 'Complete the CAPTCHA to restore your CPM to its maximum value!'),
            'solved': self.get('captcha.solved_message', 'Your CPM has been restored to its maximum value!'),
            'success': self.get('captcha.success_message', 'The longer you wait, the fewer you earn per minute')
        }
    
    def get_captcha_selector_config(self) -> Dict[str, Any]:
        """Get CAPTCHA selector configuration"""
        return {
            'button_dimensions': self.get('captcha_selector.button_dimensions', [40, 30]),
            'keypad_gap': self.get('captcha_selector.keypad_gap', [52, 42]),
            'keypad_positions': self.get('captcha_selector.keypad_positions', {
                'roc_recruit': [890, 705],
                'roc_armory': [973, 1011],
                'roc_attack': [585, 680],
                'roc_spy': [585, 695],
                'roc_training': [973, 453]
            })
        }
    
    def get_max_workers(self) -> int:
        """Get max workers"""
        return self.get('threading.max_workers', 8)
    
    def get_worker_timeout(self) -> int:
        """Get worker timeout in seconds"""
        return self.get('threading.worker_timeout_seconds', 10)
    
    def get_status_check_interval(self) -> int:
        """Get status check interval in seconds"""
        return self.get('threading.status_check_interval_seconds', 30)
    
    def get_in_progress_timeout_minutes(self) -> int:
        """Get in-progress timeout in minutes"""
        return self.get('timeouts.in_progress_timeout_minutes', 10)
    
    def get_worker_join_timeout(self) -> int:
        """Get worker join timeout in seconds"""
        return self.get('timeouts.worker_join_timeout_seconds', 10)
    
    def get_csv_file(self) -> str:
        """Get CSV file path"""
        return self.get('files.csv_file', 'accounts.csv')
    
    def get_directories(self) -> Dict[str, str]:
        """Get directory paths"""
        return {
            'error': self.get('files.error_directory', './errs'),
            'failed_captchas': self.get('files.failed_captchas_directory', './failed_captchas'),
            'low_confidence': self.get('files.low_confidence_directory', './low_confidence'),
            'correct_captchas': self.get('files.correct_captchas_directory', './correct_captchas')
        }
    
    def get_user_agent(self) -> str:
        """Get user agent string"""
        return self.get('user_agent.string', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers"""
        headers = self.get('headers', {})
        # Add user agent to headers
        headers['User-Agent'] = self.get_user_agent()
        return headers
    
    def reload(self):
        """Reload settings from file"""
        self._load_settings()
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return self._settings.copy() if self._settings else {}


# Global settings instance
_settings_instance: Optional[SettingsLoader] = None


def get_settings(settings_file: str = "settings.json") -> SettingsLoader:
    """
    Get global settings instance (singleton pattern)
    
    Args:
        settings_file: Path to settings file (only used on first call)
        
    Returns:
        SettingsLoader instance
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = SettingsLoader(settings_file)
    
    return _settings_instance


def reload_settings():
    """Reload global settings"""
    global _settings_instance
    
    if _settings_instance is not None:
        _settings_instance.reload()


if __name__ == "__main__":
    # Test the settings loader
    try:
        settings = SettingsLoader()
        
        print("Testing settings loader:")
        print(f"Database path: {settings.get_database_path()}")
        print(f"Base URL: {settings.get_base_url()}")
        print(f"Recruit URL: {settings.get_recruit_url()}")
        print(f"Login URL: {settings.get_login_url()}")
        print(f"Model path: {settings.get_model_path()}")
        print(f"Max workers: {settings.get_max_workers()}")
        print(f"Confidence threshold: {settings.get_confidence_threshold()}")
        print(f"Use CAPTCHA: {settings.get_use_captcha()}")
        print(f"CSV file: {settings.get_csv_file()}")
        print(f"Directories: {settings.get_directories()}")
        print(f"CAPTCHA messages: {settings.get_captcha_messages()}")
        print(f"CAPTCHA selector config: {settings.get_captcha_selector_config()}")
        
    except Exception as e:
        print(f"Error testing settings loader: {e}")
