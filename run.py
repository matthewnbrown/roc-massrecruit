#!/usr/bin/env python3
"""
Automated script for managing multiple accounts on roc
Converts from Selenium to requests-based approach with SQLite cookie management
"""

import sqlite3
import requests
import csv
import time
import shutil
import io
import os
import threading
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import PIL.Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from captcha_selector import ROCCaptchaSelector
from predict import CaptchaPredictor
from settings_loader import get_settings

class AccountManager:
    def __init__(self, settings_file="settings.json"):
        # Load settings
        self.settings = get_settings(settings_file)
        
        # Initialize from settings
        self.db_path = self.settings.get_database_path()
        self.base_url = self.settings.get_base_url()
        self.recruit_url = self.settings.get_recruit_url()
        self.login_url = self.settings.get_login_url()
        self.predictor = CaptchaPredictor(self.settings.get_model_path(), self.settings.get_model_device())
        self.captcha_selector = ROCCaptchaSelector()
        self.max_workers = self.settings.get_max_workers()
        self.db_lock = threading.Lock()  # Thread-safe database operations
        
        # Load CAPTCHA settings
        self.captcha_messages = self.settings.get_captcha_messages()
        self.use_captcha = self.settings.get_use_captcha()
        self.confidence_threshold = self.settings.get_confidence_threshold()
        self.max_attempts = self.settings.get_max_attempts()
        self.captcha_api_url = self.settings.get_captcha_api_url()
        
        # Load directory settings
        self.directories = self.settings.get_directories()
        
        # Load timeout settings
        self.in_progress_timeout_minutes = self.settings.get_in_progress_timeout_minutes()
        self.worker_join_timeout = self.settings.get_worker_join_timeout()
        
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing cookies and recruit solve timestamps"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_cookies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                cookies TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recruit_solves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                next_recruit_timestamp INTEGER,
                in_progress_since INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES account_cookies (username)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Check if we need to migrate from old schema

        conn.commit()
        conn.close()
    
    def save_cookies(self, username, session):
        """Save cookies from requests session to database (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert cookies to JSON-serializable format
            cookies_data = []
            for cookie in session.cookies:
                cookies_data.append({
                    'name': cookie.name,
                    'value': cookie.value,
                    'domain': cookie.domain,
                    'path': cookie.path,
                    'secure': cookie.secure
                })
            
            import json
            cookies_json = json.dumps(cookies_data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO account_cookies (username, cookies, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (username, cookies_json))
            
            conn.commit()
            conn.close()
    
    def extract_next_recruit_timestamp(self, response_text):
        """Extract the next recruit timestamp from the page HTML"""
        try:
            soup = BeautifulSoup(response_text, 'html.parser')
            countdown_element = soup.find('span', {'class': 'countdown'})
            
            if countdown_element and countdown_element.get('data-timestamp'):
                timestamp = int(countdown_element.get('data-timestamp'))
                return timestamp
            return None
        except Exception as e:
            print(f"Error extracting recruit timestamp: {e}")
            return None
    
    def save_recruit_solve_timestamp(self, username, next_recruit_timestamp=None):
        """Save the next recruit timestamp for a user (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO recruit_solves (username, next_recruit_timestamp, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (username, next_recruit_timestamp))
            
            conn.commit()
            conn.close()
    
    def mark_user_in_progress(self, username):
        """Mark a user as being processed (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_timestamp = int(time.time())
            cursor.execute('''
                INSERT OR REPLACE INTO recruit_solves (username, in_progress_since, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (username, current_timestamp))
            
            conn.commit()
            conn.close()
    
    def clear_user_in_progress(self, username):
        """Clear the in-progress status for a user (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE recruit_solves 
                SET in_progress_since = NULL, last_updated = CURRENT_TIMESTAMP
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
            conn.close()
    
    def clear_expired_in_progress(self, timeout_minutes=None):
        """Clear in-progress status for users that have been stuck for too long (thread-safe)"""
        if timeout_minutes is None:
            timeout_minutes = self.in_progress_timeout_minutes
            
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_timestamp = int(time.time())
            timeout_seconds = timeout_minutes * 60
            
            cursor.execute('''
                UPDATE recruit_solves 
                SET in_progress_since = NULL, last_updated = CURRENT_TIMESTAMP
                WHERE in_progress_since IS NOT NULL 
                AND in_progress_since < ?
            ''', (current_timestamp - timeout_seconds,))
            
            cleared_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if cleared_count > 0:
                print(f"🧹 Cleared {cleared_count} expired in-progress statuses")
    
    def can_attempt_recruit(self, username):
        """Check if user can attempt recruit based on stored timestamp (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT next_recruit_timestamp FROM recruit_solves 
                WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                # No previous solve recorded, can attempt
                return True
            
            next_recruit_timestamp = result[0]
            current_timestamp = int(time.time())
            
            # If current time is past the next recruit timestamp, can attempt
            return current_timestamp >= next_recruit_timestamp
    
    def get_eligible_users(self, csv_file):
        """Get users from CSV who are eligible to attempt recruit (5+ minutes since last solve)"""
        eligible_users = []
        
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    username = row.get('user', '').strip()
                    password = row.get('pass', '').strip()
                    email = row.get('email', '').strip()
                    
                    if username and password:
                        # Check if user is eligible to attempt recruit
                        if self.can_attempt_recruit(username):
                            eligible_users.append({
                                'username': username,
                                'password': password,
                                'email': email
                            })
                        else:
                            print(f"⏰ User {username} attempted recruit less than 5 minutes ago. Skipping.")
                    else:
                        print(f"⚠️ Skipping row with missing credentials: {row}")
                        
        except FileNotFoundError:
            print(f"❌ CSV file not found: {csv_file}")
            return []
        except Exception as e:
            print(f"❌ Error processing CSV file: {e}")
            raise e
            
        return eligible_users
    
    def sync_csv_to_database(self, csv_file):
        """Sync CSV user data to database on startup"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # clear user_credentials table
            cursor.execute('''
                DELETE FROM user_credentials
            ''')
            
            
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                synced_count = 0
                
                for row in reader:
                    username = row.get('user', '').strip()
                    password = row.get('pass', '').strip()
                    email = row.get('email', '').strip()
                    
                    if username and password:
                        # Insert or update user credentials
                        cursor.execute('''
                            INSERT OR REPLACE INTO user_credentials (username, password, email, last_updated)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (username, password, email))
                        synced_count += 1
                    else:
                        print(f"⚠️ Skipping row with missing credentials: {row}")
            
            
            # delete all rows from recruit_solves where username is not in user_credentials
            cursor.execute('''
                DELETE FROM recruit_solves WHERE username NOT IN (SELECT username FROM user_credentials)
            ''')
            
            conn.commit()
            conn.close()
            
            
            print(f"📊 Synced {synced_count} users from CSV to database")
            return synced_count
            
        except FileNotFoundError:
            print(f"❌ CSV file not found: {csv_file}")
            return 0
        except Exception as e:
            print(f"❌ Error syncing CSV to database: {e}")
            raise e
    
    def get_next_eligible_user(self):
        """Get the next eligible user from database based on recruit timestamps (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_timestamp = int(time.time())
            
            # Get users who can attempt recruit now, ordered by next recruit timestamp (earliest first)
            cursor.execute('''
                SELECT uc.username, uc.password, uc.email, 
                       COALESCE(rs.next_recruit_timestamp, 0) as next_recruit
                FROM user_credentials uc
                LEFT JOIN recruit_solves rs ON uc.username = rs.username
                WHERE COALESCE(rs.next_recruit_timestamp, 0) <= ?
                ORDER BY next_recruit ASC
                LIMIT 1
            ''', (current_timestamp,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'username': result[0],
                    'password': result[1],
                    'email': result[2]
                }
            return None
    
    def get_next_available_time(self):
        """Get the timestamp when the next user will be available (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_timestamp = int(time.time())
            
            # Get the earliest next recruit timestamp from all users
            cursor.execute('''
                SELECT MIN(COALESCE(rs.next_recruit_timestamp, 0)) as next_available
                FROM user_credentials uc
                LEFT JOIN recruit_solves rs ON uc.username = rs.username
                WHERE COALESCE(rs.next_recruit_timestamp, 0) > ?
            ''', (current_timestamp,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] and result[0] > 0:
                return result[0]
            return None
    
    def get_next_eligible_user_atomic(self):
        """Atomically get and mark the next eligible user (thread-safe)"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_timestamp = int(time.time())
            timeout_seconds = self.in_progress_timeout_minutes * 60
            
            # First, clear expired in-progress statuses
            cursor.execute('''
                UPDATE recruit_solves 
                SET in_progress_since = NULL, last_updated = CURRENT_TIMESTAMP
                WHERE in_progress_since IS NOT NULL 
                AND in_progress_since < ?
            ''', (current_timestamp - timeout_seconds,))
            
            # Get the next eligible user and mark as in-progress atomically
            cursor.execute('''
                SELECT uc.username, uc.password, uc.email
                FROM user_credentials uc
                LEFT JOIN recruit_solves rs ON uc.username = rs.username
                WHERE COALESCE(rs.next_recruit_timestamp, 0) <= ?
                AND (rs.in_progress_since IS NULL OR rs.in_progress_since < ?)
                ORDER BY COALESCE(rs.next_recruit_timestamp, 0) ASC
                LIMIT 1
            ''', (current_timestamp, current_timestamp - timeout_seconds))
            
            result = cursor.fetchone()
            
            if result:
                username = result[0]
                # Mark as in-progress
                cursor.execute('''
                    INSERT OR REPLACE INTO recruit_solves (username, in_progress_since, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (username, current_timestamp))
                
                conn.commit()
                conn.close()
                
                return {
                    'username': result[0],
                    'password': result[1],
                    'email': result[2]
                }
            
            conn.close()
            return None
    
    def streaming_worker(self, worker_id, stop_event):
        """Continuous worker that pulls users as they become available"""
        print(f"🧵 Worker {worker_id} started")
        
        while not stop_event.is_set():
            try:
                # Get next user atomically (already marked as in-progress)
                user_data = self.get_next_eligible_user_atomic()
                
                if user_data:
                    username = user_data['username']
                    #rint(f"🧵 [Worker {worker_id}] Processing: {username}")
                    
                    success = self.process_account(
                        user_data['username'], 
                        user_data['password'], 
                        user_data['email']
                    )
                    
                    if not success:
                        print(f"❌ [Worker {worker_id}] Failed: {username}")
                    
                    # Clear in-progress status
                    self.clear_user_in_progress(username)
                else:
                    # No users available, wait a bit
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ [Worker {worker_id}] Error: {e}")
                time.sleep(5)  # Wait longer on error
        
        print(f"🧵 Worker {worker_id} stopped")
    
    def load_cookies(self, username, session):
        """Load cookies from database into requests session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT cookies FROM account_cookies WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            import json
            cookies_data = json.loads(result[0])
            
            for cookie_data in cookies_data:
                session.cookies.set(
                    cookie_data['name'],
                    cookie_data['value'],
                    domain=cookie_data['domain'],
                    path=cookie_data['path']
                )
            return True
        return False
    
    def is_logged_in(self, session, load_page=True):
        """Check if user is logged in by looking for login_form on recruit.php"""
        try:
            if load_page:
                response = session.get(self.recruit_url)
                response.raise_for_status()
            
            return response.text.find('placeholder="email@address.com') == -1
        except Exception as e:
            print(f"Error checking login status: {e}")
            return False
    
    def login(self, username, email, password, session):
        """Login using requests"""
        try:
            # First, get the login page to extract any CSRF tokens or form data
            login_page = session.get(self.base_url)
            
            # Find the login form
            if login_page.text.find('placeholder="email@address.com') == -1:
                print("Could not find login form... Are we already logged in?")
                return False
            
            payload = {"email": email, "password": password}

            
            # Submit login form
            response = session.post(self.login_url, payload)
            response.raise_for_status()
            
            # Check if login was successful
            if response.text.find('placeholder="email@address.com') != -1:
                print(f"Login failed for {email}")
                return False
            else:
                print(f"Login successful for {email}")
                # Save cookies after successful login
                self.save_cookies(username, session)
                return True
                
        except Exception as e:
            print(f"Error during login for {email}: {e}")
            return False
    
    def recruit(self, session, username):
        """Solve CAPTCHA using the existing predictor"""
        for attempt in range(self.max_attempts):
            # Get the recruit page
            use_captcha = self.use_captcha
            
            unsolved_captcha_str = self.captcha_messages['unsolved']
            solved_captcha_str = self.captcha_messages['solved']         

            if use_captcha:
                response = session.get(self.recruit_url)
                response.raise_for_status()

                # Skip if captcha is not needed:
                if unsolved_captcha_str not in response.text:
                    # Extract and save the next recruit timestamp from the page
                    next_timestamp = self.extract_next_recruit_timestamp(response.text)
                    if next_timestamp:
                        self.save_recruit_solve_timestamp(username, next_timestamp)
                        clocktime = datetime.fromtimestamp(next_timestamp).strftime('%H:%M:%S')
                        print(f"✅ {username} | No CAPTCHA needed. Skipping. Next recruit available at timestamp: {clocktime}")
                    return True
                
                # print(f"\n🔁 Attempt {attempt + 1} to solve CAPTCHA for {username}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                captcha_img = soup.find('img', {'id': 'captcha_image'})
                
                if not captcha_img:
                    os.makedirs(self.directories['error'], exist_ok=True)
                    with open(f"{self.directories['error']}/no_captcha_{username}.html", 'w', encoding='utf-16') as file:
                        file.write(response.text)
                    print("⚠️ No CAPTCHA found. Skipping.")
                    continue
                
                captcha_url = captcha_img.get('src')
                if not captcha_url:
                    print("⚠️ No CAPTCHA URL found.")
                    continue
                
                captcha_url = urljoin(self.base_url, captcha_url)
                # Extract hash from URL
                hash_value = captcha_url.split('hash=')[1] if 'hash=' in captcha_url else 'unknown'
                # Download CAPTCHA image
                captcha_response = session.get(captcha_url)
                captcha_name = f'captcha_{hash_value}.png'
                
            
                if captcha_response.status_code == 200:
                    img = PIL.Image.open(io.BytesIO(captcha_response.content))
                    path = captcha_name
                    img.save(path)
                else:
                    print("Error downloading CAPTCHA, non 200 code")
                    continue
                # Use API to solve CAPTCHA
                try:
                    with open(captcha_name, 'rb') as img_file:
                        files = {
                            'image': ('captcha.png', img_file, 'image/png')
                        }
                        data = {
                            'captcha_hash': hash_value
                        }
                    
                        # Make API request to solve captcha
                        api_response = requests.post(self.captcha_api_url, files=files, data=data)
                        
                        if api_response.status_code != 200:
                            print(f"❌ API request failed: {api_response.text}")
                            continue
                        
                        resp_json = api_response.json()
                        num = resp_json['predicted_answer']
                        confidence = resp_json['confidence']
                        
                        
                        
                except Exception as e:
                    print(f"❌ API error: {e}")
                    continue
                
                if confidence < self.confidence_threshold:
                    print(f"⚠️ {username} | confidence too low ({num}|{confidence}). Skipping guess.")
                    os.makedirs(self.directories['low_confidence'], exist_ok=True)
                    shutil.move(captcha_name, f"{self.directories['low_confidence']}/{num}_{hash_value}.png")
                    
                    continue
                
                x,y = self.captcha_selector.get_xy_static(num, 'roc_recruit')
                
                captcha_payload = {
                    "num": num,
                    "captcha": hash_value,
                    "coordinates[x]": x,
                    "coordinates[y]": y
                }
            else:
                captcha_payload = { "submit": "Recruit" }

            # Submit CAPTCHA solution
            try:
                button_response = session.post(self.recruit_url, captcha_payload)
                button_response.raise_for_status()

                # Check if CAPTCHA was solved
                if use_captcha and unsolved_captcha_str in button_response.text:
                    print(f"❌ {username} | CAPTCHA failed ({captcha_name}). Retrying...")
                    os.makedirs(self.directories['failed_captchas'], exist_ok=True)
                    shutil.move(captcha_name, f"{self.directories['failed_captchas']}/{num}_{hash_value}.png")
                    continue
                elif self.captcha_messages['success'] in button_response.text:
                    # Save updated cookies
                    self.save_cookies(username, session)
                    # Extract and save the next recruit timestamp from the response
                    next_timestamp = self.extract_next_recruit_timestamp(button_response.text)
                    if next_timestamp:
                        self.save_recruit_solve_timestamp(username, next_timestamp)
                        clocktime = datetime.fromtimestamp(next_timestamp).strftime('%H:%M:%S')
                        print(f"✅ {username} | Next recruit available at timestamp: {clocktime}")
                        
                    if use_captcha:
                        os.makedirs(self.directories['correct_captchas'], exist_ok=True)
                        shutil.move(captcha_name, f"{self.directories['correct_captchas']}/{num}_{hash_value}.png")
                    return True
                elif solved_captcha_str not in button_response.text:
                    print(f"❌ {username} | Recruit failed ({captcha_name}). Retrying...")
            except Exception as e:
                print(f"❌ Button click error: {e}")
                continue
        
        print("🚫 All CAPTCHA attempts failed.")
        return False
    
    def process_account(self, username, password, email):
        """Process a single account"""        
        # Create a new session for this account
        session = requests.Session()
        session.headers.update(self.settings.get_headers())
        
        # Try to load existing cookies
        cookies_loaded = self.load_cookies(username, session)
        
        # Check if logged in
        if not self.is_logged_in(session):
            print(f"🔑 User {username} is logged out, attempting login...")
            if not self.login(username, email, password, session):
                print(f"❌ Failed to login {username}")
                return False
        else:
            pass
            #print(f"✅ User {username} is already logged in")
        
        # Try to solve CAPTCHA (user eligibility already checked in get_eligible_users)
        success = self.recruit(session, username)
        if not success:
            print(f"❌ CAPTCHA process failed for {username}")
        
        return success
    
    def process_accounts_from_csv(self, csv_file):
        """Process multiple accounts from CSV file - only eligible users (5+ minutes since last solve)"""
        # Get only eligible users (haven't solved in 5+ minutes)
        eligible_users = self.get_eligible_users(csv_file)
        
        if not eligible_users:
            print("📋 No eligible users found (all users attempted recruit within last 5 minutes)")
            return
        
        print(f"📋 Found {len(eligible_users)} eligible users to process")
        
        # Process each eligible user
        for user_data in eligible_users:
            self.process_account(user_data['username'], user_data['password'], user_data['email'])

def main():
    """Main function - continuously processes eligible users from database"""
    manager = AccountManager()
    
    # Check if CSV file exists and sync to database
    csv_file = manager.settings.get_csv_file()
    if os.path.exists(csv_file):
        print(f"📁 Found CSV file: {csv_file}")
        print("🔄 Syncing CSV data to database...")
        synced_count = manager.sync_csv_to_database(csv_file)
        
        if synced_count == 0:
            print("❌ No users synced from CSV. Exiting.")
            return
        
        print("🚀 Starting streaming multithreaded processing...")
        print(f"🧵 Using {manager.max_workers} continuous worker threads")
        print("Press Ctrl+C to stop")
        
        # Create stop event for graceful shutdown
        stop_event = threading.Event()
        workers = []
        
        try:
            # Start worker threads
            for i in range(manager.max_workers):
                worker = threading.Thread(
                    target=manager.streaming_worker, 
                    args=(i + 1, stop_event),
                    daemon=True
                )
                worker.start()
                workers.append(worker)
            
            print(f"✅ Started {manager.max_workers} worker threads")
            
            # Main monitoring loop
            while True:
                # Check if any workers are still alive
                alive_workers = [w for w in workers if w.is_alive()]
                
                if not alive_workers:
                    print("❌ All workers have died. Exiting.")
                    break
                
                # Show status every configured interval
                time.sleep(manager.settings.get_status_check_interval())
                next_available_time = manager.get_next_available_time()
                if next_available_time:
                    current_time = int(time.time())
                    wait_seconds = next_available_time - current_time
                    
                    if wait_seconds > 0:
                        wait_minutes = wait_seconds // 60
                        wait_remaining_seconds = wait_seconds % 60
                        available_time = datetime.fromtimestamp(next_available_time).strftime('%H:%M:%S')
                        
                        if wait_minutes > 0:
                            print(f"📊 Status: {len(alive_workers)} workers active. Next user available at {available_time} ({wait_minutes}m {wait_remaining_seconds}s)")
                        else:
                            print(f"📊 Status: {len(alive_workers)} workers active. Next user available at {available_time} ({wait_seconds}s)")
                    else:
                        print(f"📊 Status: {len(alive_workers)} workers active. Users available now.")
                else:
                    print(f"📊 Status: {len(alive_workers)} workers active. No users with timestamps found.")
                    
        except KeyboardInterrupt:
            print("\n🛑 Script stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Graceful shutdown
            print("🛑 Stopping all workers...")
            stop_event.set()
            
            # Wait for workers to finish (with timeout)
            for worker in workers:
                worker.join(timeout=manager.worker_join_timeout)
            
            print("✅ All workers stopped")
            
    else:
        print(f"⚠️ CSV file not found: {csv_file}")
        print("Creating example CSV file...")
        
        # Create example CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['user', 'pass', 'email'])
            writer.writerow(['example_user', 'example_pass', 'example@email.com'])
        
        print(f"✅ Created example CSV file: {csv_file}")
        print("Please edit the CSV file with your actual account credentials and run the script again.")

if __name__ == "__main__":
    main()
