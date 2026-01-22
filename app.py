# receipt_reward_system.py - COMPLETE DUAL ENVIRONMENT VERSION (Local + Render)
# Admin access: Add ?admin=1&password=your_admin_password to URL
# Local: http://127.0.0.1:5000
# Render: https://your-app.onrender.com

import os
import requests
import random
import string
import imagehash
import hashlib
import uuid
import sys
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import re
import json
import subprocess
from urllib.parse import urlparse

# ========== ENVIRONMENT CONFIGURATION ==========
# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load .env file for local development

# Telegram Configuration - FROM ENVIRONMENT VARIABLES
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_BOT_USERNAME = os.environ.get('TELEGRAM_BOT_USERNAME', '')

# Admin Configuration
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')  # Change in production

# ========== TESSERACT PATH FINDER ==========
def find_tesseract_path():
    """Find Tesseract installation path"""
    possible_paths = []
    
    # Windows paths
    if sys.platform == 'win32':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe'),
            r'C:\Users\DELL\AppData\Local\Tesseract-OCR\tesseract.exe',
        ]
        
        # Try to find via command line
        try:
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout.strip():
                paths = result.stdout.strip().split('\n')
                possible_paths = paths + possible_paths
        except:
            pass
    
    # Linux paths (Render uses Linux)
    elif sys.platform.startswith('linux'):
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/bin/tesseract',
            '/app/.apt/usr/bin/tesseract',  # Render specific
        ]
    
    # macOS paths
    elif sys.platform == 'darwin':
        possible_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract',
        ]
    
    # Check if any path exists
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Last resort: try command line
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    return None

# ========== OCR SETUP ==========
OCR_AVAILABLE = False
tesseract_path = None

try:
    import pytesseract
    from PIL import Image
    
    # Find Tesseract path
    tesseract_path = find_tesseract_path()
    
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✓ Tesseract found at: {tesseract_path}")
        
        # Test OCR
        try:
            # Create test image
            img = Image.new('RGB', (100, 50), color='white')
            text = pytesseract.image_to_string(img)
            OCR_AVAILABLE = True
            print("✓ OCR test successful")
        except Exception as e:
            print(f"✗ OCR test failed: {e}")
            OCR_AVAILABLE = False
    else:
        print("✗ Tesseract not found")
        OCR_AVAILABLE = False
        
except ImportError:
    print("✗ pytesseract not installed. Run: pip install pytesseract")
    OCR_AVAILABLE = False
except Exception as e:
    print(f"✗ OCR setup error: {e}")
    OCR_AVAILABLE = False

# ========== TELEGRAM FUNCTIONS ==========
def send_telegram_message(chat_id, message):
    """Send message via Telegram bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ Telegram bot token not configured")
        return False
        
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

def generate_verification_code():
    """Generate 6-digit verification code"""
    return ''.join(random.choices(string.digits, k=6))

# ========== FLASK APP SETUP ==========
app = Flask(__name__)

# Database Configuration - DUAL ENVIRONMENT
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # Render PostgreSQL
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print("✓ Using PostgreSQL database (Render)")
else:
    # Local SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///receipts.db'
    print("✓ Using SQLite database (Local)")

# Other Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'gif', 'bmp'}

# File Upload Configuration - DUAL ENVIRONMENT
if os.environ.get('RENDER'):  # Running on Render
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'  # Render's ephemeral storage
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    print("✓ Using Render file storage (/tmp/uploads/)")
else:
    app.config['UPLOAD_FOLDER'] = 'uploads/'  # Local storage
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    print("✓ Using local file storage (uploads/)")

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# ========== DATABASE MODELS ==========
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    points = db.Column(db.Integer, default=0)
    
    # Use string reference for relationship - IMPORTANT!
    receipts = db.relationship('Receipt', backref='user', lazy=True, cascade='all, delete-orphan')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Telegram verification fields
    is_verified = db.Column(db.Boolean, default=False)
    telegram_id = db.Column(db.String(50), unique=True, nullable=True)
    verification_code = db.Column(db.String(6), nullable=True)
    verification_expiry = db.Column(db.DateTime, nullable=True)
    verified_at = db.Column(db.DateTime, nullable=True)

# ========== RECEIPT MODEL ==========
class Receipt(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    ocr_text = db.Column(db.Text)
    extracted_amount = db.Column(db.Float)
    verified_amount = db.Column(db.Float)
    points_awarded = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='pending')
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed_date = db.Column(db.DateTime)
    
    # Duplicate detection fields
    image_hash = db.Column(db.String(64), nullable=True)
    content_hash = db.Column(db.String(64), nullable=True)
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of = db.Column(db.String(36), db.ForeignKey('receipt.id'), nullable=True)

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ========== OCR PROCESSOR ==========
class ReceiptOCRProcessor:
    def __init__(self):
        self.use_ocr = OCR_AVAILABLE
        
    def preprocess_image(self, image_path: str):
        """Preprocess image for better OCR results"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            denoised = cv2.fastNlMeansDenoising(gray)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return None
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from receipt image"""
        if not self.use_ocr:
            return "OCR_NOT_AVAILABLE"
        
        try:
            # Try direct OCR first
            text = pytesseract.image_to_string(Image.open(image_path))
            if text.strip():
                return text
            
            # Try with preprocessing
            processed_img = self.preprocess_image(image_path)
            if processed_img is not None:
                # Convert numpy array to PIL Image
                pil_img = Image.fromarray(processed_img)
                text = pytesseract.image_to_string(pil_img)
            
            return text if text else ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_total_amount(self, text: str) -> float:
        """Extract total amount from OCR text"""
        if not text or text == "OCR_NOT_AVAILABLE":
            return 0.0
        
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Patterns for Malaysian receipts
        patterns = [
            r'(?:total|jumlah|amount)[\s:]*rm?\s*([\d,]+\.?\d{0,2})',
            r'rm?\s*([\d,]+\.?\d{2})\s*$',
            r'rm?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'rm\s*([\d\.]+)',
            r'(\d+\.\d{2})\s*(?!\d*\.\d)'
        ]
        
        all_amounts = []
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    amount_str = str(match).replace(',', '').replace(' ', '')
                    try:
                        amount = float(amount_str)
                        if 0.1 <= amount <= 100000:
                            all_amounts.append(amount)
                    except ValueError:
                        continue
            except:
                continue
        
        if all_amounts:
            return max(all_amounts)
        
        # Fallback: find all numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        amounts = []
        for num in numbers:
            try:
                val = float(num)
                if 1 <= val <= 100000:
                    amounts.append(val)
            except:
                continue
        
        return max(amounts) if amounts else 0.0
    
    def extract_receipt_info(self, text: str) -> dict:
        """Extract receipt information"""
        info = {
            'total_amount': 0.0,
            'date': None,
            'store_name': None,
            'raw_text': text[:500]
        }
        
        info['total_amount'] = self.extract_total_amount(text)
        
        # Extract date
        date_patterns = [
            r'\d{2}[/-]\d{2}[/-]\d{4}',
            r'\d{4}[/-]\d{2}[/-]\d{2}',
            r'\d{2}\s+[A-Za-z]+\s+\d{4}',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                info['date'] = matches[0]
                break
        
        return info

# Create OCR processor instance
ocr_processor = ReceiptOCRProcessor()

# ========== DUPLICATE DETECTOR ==========
class DuplicateDetector:
    def __init__(self, hamming_threshold=5):
        self.hamming_threshold = hamming_threshold
    
    def generate_image_hash(self, image_path):
        """Generate perceptual hash for image"""
        try:
            img = Image.open(image_path)
            img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            hash_obj = imagehash.average_hash(img)
            return str(hash_obj)
        except Exception as e:
            print(f"Image hash generation error: {e}")
            return None
    
    def generate_content_hash(self, ocr_text, amount, user_id):
        """Generate hash from OCR content"""
        if not ocr_text or ocr_text == "OCR_NOT_AVAILABLE":
            return None
        
        try:
            import re
            clean_text = ocr_text.lower().replace('\n', ' ').replace('\r', ' ')
            
            # Extract store name
            store_patterns = [
                r'([a-z\s&]+)(?:\s*(?:sdn|bhd|store|market|supermarket|mall|plaza))',
                r'from:\s*([^\n]+)',
                r'merchant:\s*([^\n]+)',
                r'store:\s*([^\n]+)'
            ]
            
            store_name = None
            for pattern in store_patterns:
                match = re.search(pattern, clean_text)
                if match:
                    store_name = match.group(1).strip()[:50]
                    break
            
            # Extract date
            date_patterns = [
                r'\d{2}[/-]\d{2}[/-]\d{4}',
                r'\d{4}[/-]\d{2}[/-]\d{2}',
                r'\d{1,2}\s+[a-z]+\s+\d{4}',
            ]
            
            receipt_date = None
            for pattern in date_patterns:
                match = re.search(pattern, clean_text)
                if match:
                    receipt_date = match.group(0)
                    break
            
            # Create fingerprint
            fingerprint_parts = [
                str(user_id),
                f"{float(amount):.2f}" if amount else "0.00",
                store_name or "unknown_store",
                receipt_date or "unknown_date",
                hashlib.md5(clean_text[:100].encode()).hexdigest()[:8]
            ]
            
            fingerprint = "|".join(fingerprint_parts)
            content_hash = hashlib.md5(fingerprint.encode()).hexdigest()
            return content_hash
            
        except Exception as e:
            print(f"Content hash generation error: {e}")
            return None
    
    def check_duplicate(self, user_id, image_path, ocr_text=None, amount=None):
        """Check if receipt is a duplicate"""
        duplicates_found = []
        
        # 1. Check by image hash
        image_hash = self.generate_image_hash(image_path)
        if image_hash:
            similar_receipts = Receipt.query.filter(
                Receipt.user_id == user_id,
                Receipt.image_hash.isnot(None)
            ).all()
            
            for receipt in similar_receipts:
                if receipt.image_hash:
                    hamming_dist = self.hamming_distance(image_hash, receipt.image_hash)
                    if hamming_dist <= self.hamming_threshold:
                        duplicates_found.append({
                            'receipt_id': receipt.id,
                            'filename': receipt.filename,
                            'match_type': 'image_similarity',
                            'confidence': 'high',
                            'hamming_distance': hamming_dist
                        })
        
        # 2. Check by content hash
        if ocr_text and amount:
            content_hash = self.generate_content_hash(ocr_text, amount, user_id)
            if content_hash:
                duplicate_receipts = Receipt.query.filter(
                    Receipt.user_id == user_id,
                    Receipt.content_hash == content_hash,
                    Receipt.is_duplicate == False
                ).all()
                
                for receipt in duplicate_receipts:
                    duplicates_found.append({
                        'receipt_id': receipt.id,
                        'filename': receipt.filename,
                        'match_type': 'content_match',
                        'confidence': 'very_high',
                        'amount': receipt.extracted_amount,
                        'upload_date': receipt.upload_date
                    })
        
        return duplicates_found if duplicates_found else None
    
    def hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        if len(hash1) != len(hash2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Create duplicate detector instance
duplicate_detector = DuplicateDetector()

# ========== API ROUTES ==========
@app.route('/api/register', methods=['POST'])
def register_user():
    """Register new user with Telegram verification"""
    data = request.get_json() if request.is_json else request.form
    if not data or 'username' not in data or 'telegram_id' not in data:
        return jsonify({'error': 'Missing required fields (username and telegram_id)'}), 400
    
    try:
        telegram_id = data['telegram_id'].strip()
        username = data['username'].strip()
        
        # Check if user exists
        existing = User.query.filter(
            (User.username == username) | 
            (User.telegram_id == telegram_id)
        ).first()
        
        if existing:
            return jsonify({'error': 'Username or Telegram ID already exists'}), 400
        
        # Generate verification code
        verification_code = generate_verification_code()
        expiry_time = datetime.utcnow() + timedelta(hours=24)
        
        # Create user (unverified)
        user = User(
            username=username,
            email=f"{username}@telegram.user",
            points=0,
            is_verified=False,
            telegram_id=telegram_id,
            verification_code=verification_code,
            verification_expiry=expiry_time
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Send verification code via Telegram
        message = f"""
✅ <b>Receipt Reward System Verification</b>

Hello {username}!

Your verification code is:
<b><code>{verification_code}</code></b>

Send this code back to me to verify your account.
This code expires in 24 hours.

Go to the web app to enter this code and complete verification.
        """
        
        if send_telegram_message(telegram_id, message):
            return jsonify({
                'message': 'User registered. Please check Telegram for verification code.',
                'user_id': user.id,
                'username': user.username,
                'requires_verification': True,
                'verification_method': 'telegram'
            }), 201
        else:
            db.session.delete(user)
            db.session.commit()
            return jsonify({
                'error': 'Failed to send Telegram verification. Check your Telegram ID.'
            }), 400
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/verify', methods=['POST'])
def verify_user():
    """Verify user with Telegram code"""
    data = request.get_json()
    if not data or 'user_id' not in data or 'verification_code' not in data:
        return jsonify({'error': 'Missing user_id or verification_code'}), 400
    
    try:
        user_id = data['user_id']
        code = data['verification_code'].strip()
        
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if user.is_verified:
            return jsonify({'error': 'User already verified'}), 400
        
        if user.verification_code != code:
            return jsonify({'error': 'Invalid verification code'}), 400
        
        if datetime.utcnow() > user.verification_expiry:
            return jsonify({'error': 'Verification code expired'}), 400
        
        # Verify user
        user.is_verified = True
        user.verified_at = datetime.utcnow()
        user.verification_code = None
        user.verification_expiry = None
        
        db.session.commit()
        
        # Send confirmation via Telegram
        success_message = f"""
🎉 <b>Account Verified Successfully!</b>

Your account <b>{user.username}</b> is now verified.

You can now:
• Upload receipts to earn points
• Track your reward points
• Redeem rewards when available

Start by uploading your first receipt!
        """
        
        send_telegram_message(user.telegram_id, success_message)
        
        return jsonify({
            'message': 'Account verified successfully!',
            'user_id': user.id,
            'username': user.username,
            'is_verified': True,
            'verified_at': user.verified_at.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/user/<int:user_id>/status', methods=['GET'])
def get_user_status(user_id):
    """Get user verification status"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user_id': user.id,
        'username': user.username,
        'is_verified': user.is_verified,
        'points': user.points,
        'telegram_id': user.telegram_id,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'verified_at': user.verified_at.isoformat() if user.verified_at else None,
        'receipts_count': len(user.receipts)
    })

@app.route('/api/upload', methods=['POST'])
def upload_receipt():
    """Upload receipt - requires verified user"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    # Verify user exists AND is verified
    user = db.session.get(User, int(user_id)) if user_id.isdigit() else None
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if not user.is_verified:
        return jsonify({
            'error': 'Account not verified. Please verify your Telegram account first.',
            'requires_verification': True,
            'user_id': user.id
        }), 403
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{file_id}.{file_extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save file
    file.save(file_path)
    
    # Create receipt record
    receipt = Receipt(
        id=file_id,
        user_id=user.id,
        filename=filename,
        file_path=file_path,
        status='processing'
    )
    db.session.add(receipt)
    db.session.commit()
    
    # Process receipt
    process_receipt_sync(receipt.id)
    
    return jsonify({
        'message': 'Receipt uploaded successfully',
        'receipt_id': receipt.id,
        'status': 'processing'
    }), 202

def process_receipt_sync(receipt_id: str):
    """Process receipt OCR with duplicate detection"""
    session = Session(db.engine)
    try:
        receipt = session.get(Receipt, receipt_id)
        if not receipt:
            return
        
        print(f"Processing receipt: {receipt_id}")
        
        # Extract text
        ocr_text = ocr_processor.extract_text(receipt.file_path)
        receipt.ocr_text = ocr_text
        
        # Extract amount
        receipt_info = ocr_processor.extract_receipt_info(ocr_text)
        extracted_amount = receipt_info.get('total_amount', 0.0)
        
        print(f"Extracted amount: {extracted_amount}")
        
        # Generate image hash for duplicate detection
        image_hash = duplicate_detector.generate_image_hash(receipt.file_path)
        receipt.image_hash = image_hash
        
        # Check for duplicates BEFORE awarding points
        duplicates = duplicate_detector.check_duplicate(
            user_id=receipt.user_id,
            image_path=receipt.file_path,
            ocr_text=ocr_text,
            amount=extracted_amount
        )
        
        if duplicates:
            print(f"⚠️ Duplicate detected for receipt {receipt_id}")
            receipt.is_duplicate = True
            
            if duplicates:
                original_receipt = session.get(Receipt, duplicates[0]['receipt_id'])
                if original_receipt and not original_receipt.is_duplicate:
                    receipt.duplicate_of = original_receipt.id
            
            receipt.status = 'duplicate'
            receipt.points_awarded = 0
            
            # Generate content hash
            content_hash = duplicate_detector.generate_content_hash(
                ocr_text, extracted_amount, receipt.user_id
            )
            receipt.content_hash = content_hash
            
            # Send Telegram notification
            user = session.get(User, receipt.user_id)
            if user and user.telegram_id:
                message = f"""
⚠️ <b>Duplicate Receipt Detected</b>

Your receipt "{receipt.filename}" appears to be a duplicate of:
<b>{duplicates[0]['filename']}</b>

Match type: {duplicates[0]['match_type']}
Confidence: {duplicates[0]['confidence']}

No points were awarded for this duplicate receipt.
                """
                send_telegram_message(user.telegram_id, message)
                
        elif extracted_amount and extracted_amount > 0:
            # Not a duplicate - process normally
            receipt.extracted_amount = extracted_amount
            receipt.verified_amount = extracted_amount
            receipt.points_awarded = int(extracted_amount)
            receipt.status = 'processed'
            
            # Generate content hash
            content_hash = duplicate_detector.generate_content_hash(
                ocr_text, extracted_amount, receipt.user_id
            )
            receipt.content_hash = content_hash
            
            # Update user points
            user = session.get(User, receipt.user_id)
            if user:
                user.points += receipt.points_awarded
                print(f"Awarded {receipt.points_awarded} points to user {user.id}")
                
                # Send Telegram notification
                if user.telegram_id:
                    message = f"""
🎉 <b>Receipt Processed Successfully!</b>

Receipt: <b>{receipt.filename}</b>
Amount: RM <b>{extracted_amount:.2f}</b>
Points awarded: <b>{receipt.points_awarded}</b>

Your total points: <b>{user.points}</b>
                    """
                    send_telegram_message(user.telegram_id, message)
        else:
            receipt.status = 'failed'
            receipt.points_awarded = 0
            print(f"No amount extracted from receipt {receipt_id}")
        
        receipt.processed_date = datetime.utcnow()
        session.commit()
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        session.rollback()
        try:
            receipt = session.get(Receipt, receipt_id)
            if receipt:
                receipt.status = 'error'
                session.commit()
        except:
            pass
    finally:
        session.close()

@app.route('/api/user/<int:user_id>/points', methods=['GET'])
def get_user_points(user_id):
    """Get user points"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user_id': user.id,
        'username': user.username,
        'points': user.points,
        'receipts_count': len(user.receipts),
        'is_verified': user.is_verified
    })

@app.route('/api/receipt/<receipt_id>', methods=['GET'])
def get_receipt_status(receipt_id):
    """Get receipt status"""
    receipt = db.session.get(Receipt, receipt_id)
    if not receipt:
        return jsonify({'error': 'Receipt not found'}), 404
    
    return jsonify({
        'receipt_id': receipt.id,
        'user_id': receipt.user_id,
        'status': receipt.status,
        'extracted_amount': receipt.extracted_amount,
        'points_awarded': receipt.points_awarded,
        'upload_date': receipt.upload_date.isoformat() if receipt.upload_date else None,
        'processed_date': receipt.processed_date.isoformat() if receipt.processed_date else None
    })

@app.route('/api/user/<int:user_id>/receipts', methods=['GET'])
def get_user_receipts(user_id):
    """Get user receipts"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    receipts = Receipt.query.filter_by(user_id=user_id).order_by(Receipt.upload_date.desc()).all()
    
    receipt_list = []
    for receipt in receipts:
        receipt_list.append({
            'id': receipt.id,
            'filename': receipt.filename,
            'status': receipt.status,
            'amount': receipt.extracted_amount,
            'points': receipt.points_awarded,
            'upload_date': receipt.upload_date.isoformat() if receipt.upload_date else None,
            'processed_date': receipt.processed_date.isoformat() if receipt.processed_date else None
        })
    
    return jsonify({
        'user_id': user.id,
        'username': user.username,
        'total_points': user.points,
        'total_receipts': len(receipts),
        'receipts': receipt_list
    })

@app.route('/api/receipt/<receipt_id>/manual_amount', methods=['POST'])
def manual_amount_entry(receipt_id):
    """Manual amount entry"""
    data = request.get_json()
    if not data or 'amount' not in data:
        return jsonify({'error': 'Amount required'}), 400
    
    try:
        amount = float(data['amount'])
        if amount <= 0:
            return jsonify({'error': 'Amount must be positive'}), 400
        
        receipt = db.session.get(Receipt, receipt_id)
        if not receipt:
            return jsonify({'error': 'Receipt not found'}), 404
        
        if receipt.status not in ['failed', 'error']:
            return jsonify({'error': 'Can only update failed receipts'}), 400
        
        # Update receipt
        receipt.verified_amount = amount
        receipt.extracted_amount = amount
        receipt.points_awarded = int(amount)
        receipt.status = 'verified'
        receipt.processed_date = datetime.utcnow()
        
        # Update user points
        user = db.session.get(User, receipt.user_id)
        user.points += receipt.points_awarded
        
        db.session.commit()
        
        return jsonify({
            'message': 'Receipt manually verified',
            'points_awarded': receipt.points_awarded,
            'new_user_points': user.points
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid amount'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/receipt/<receipt_id>/check_duplicate', methods=['GET'])
def check_duplicate_status(receipt_id):
    """Check if a receipt is a duplicate"""
    receipt = db.session.get(Receipt, receipt_id)
    if not receipt:
        return jsonify({'error': 'Receipt not found'}), 404
    
    if not receipt.is_duplicate:
        return jsonify({
            'is_duplicate': False,
            'message': 'Receipt is not a duplicate'
        })
    
    original_receipt = None
    if receipt.duplicate_of:
        original_receipt = db.session.get(Receipt, receipt.duplicate_of)
    
    return jsonify({
        'is_duplicate': True,
        'receipt_id': receipt.id,
        'duplicate_of': receipt.duplicate_of,
        'original_filename': original_receipt.filename if original_receipt else None,
        'original_amount': original_receipt.extracted_amount if original_receipt else None,
        'original_upload_date': original_receipt.upload_date.isoformat() if original_receipt else None,
        'points_awarded': receipt.points_awarded
    })

# ========== ADMIN API ROUTES ==========
@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    """Get all users (admin only)"""
    # Admin check via password parameter
    admin_key = request.args.get('admin_key')
    if admin_key != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    
    users = User.query.all()
    user_list = []
    
    for user in users:
        user_list.append({
            'id': user.id,
            'username': user.username,
            'telegram_id': user.telegram_id,
            'points': user.points,
            'is_verified': user.is_verified,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'verified_at': user.verified_at.isoformat() if user.verified_at else None,
            'receipts_count': len(user.receipts)
        })
    
    return jsonify({
        'total_users': len(user_list),
        'users': user_list
    })

@app.route('/api/admin/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics (admin only)"""
    # Admin check
    admin_key = request.args.get('admin_key')
    if admin_key != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    
    total_users = User.query.count()
    total_verified_users = User.query.filter_by(is_verified=True).count()
    total_receipts = Receipt.query.count()
    total_points_awarded = db.session.query(db.func.sum(User.points)).scalar() or 0
    
    # Get recent activity (last 24 hours)
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    recent_receipts = Receipt.query.filter(Receipt.upload_date >= one_day_ago).order_by(Receipt.upload_date.desc()).limit(20).all()
    
    recent_activity = []
    for receipt in recent_receipts:
        recent_activity.append({
            'id': receipt.id,
            'user_id': receipt.user_id,
            'username': receipt.user.username if receipt.user else 'Unknown',
            'filename': receipt.filename,
            'amount': receipt.extracted_amount,
            'points': receipt.points_awarded,
            'status': receipt.status,
            'upload_date': receipt.upload_date.isoformat() if receipt.upload_date else None
        })
    
    # Get top users by points
    top_users = User.query.filter(User.points > 0).order_by(User.points.desc()).limit(10).all()
    leaderboard = []
    for user in top_users:
        leaderboard.append({
            'username': user.username,
            'points': user.points,
            'receipts_count': len(user.receipts)
        })
    
    return jsonify({
        'total_users': total_users,
        'total_verified_users': total_verified_users,
        'total_receipts': total_receipts,
        'total_points_awarded': total_points_awarded,
        'recent_activity': recent_activity,
        'leaderboard': leaderboard
    })

@app.route('/api/admin/user/<int:user_id>/details', methods=['GET'])
def get_admin_user_details(user_id):
    """Get detailed user info for admin"""
    admin_key = request.args.get('admin_key')
    if admin_key != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get all receipts for this user
    receipts = Receipt.query.filter_by(user_id=user_id).order_by(Receipt.upload_date.desc()).all()
    
    receipt_list = []
    for receipt in receipts:
        receipt_list.append({
            'id': receipt.id,
            'filename': receipt.filename,
            'status': receipt.status,
            'amount': receipt.extracted_amount,
            'points': receipt.points_awarded,
            'is_duplicate': receipt.is_duplicate,
            'upload_date': receipt.upload_date.isoformat() if receipt.upload_date else None,
            'processed_date': receipt.processed_date.isoformat() if receipt.processed_date else None
        })
    
    return jsonify({
        'user': {
            'id': user.id,
            'username': user.username,
            'telegram_id': user.telegram_id,
            'email': user.email,
            'points': user.points,
            'is_verified': user.is_verified,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'verified_at': user.verified_at.isoformat() if user.verified_at else None
        },
        'receipts': receipt_list,
        'total_receipts': len(receipts),
        'total_points_earned': sum(r.points_awarded for r in receipts if r.points_awarded)
    })

# ========== DEBUG ENDPOINTS ==========
@app.route('/api/debug/ocr', methods=['GET'])
def debug_ocr():
    """Debug OCR status"""
    info = {
        'ocr_available': OCR_AVAILABLE,
        'tesseract_path': tesseract_path,
        'platform': sys.platform,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'environment': 'Render' if os.environ.get('RENDER') else 'Local',
        'database': 'PostgreSQL' if os.environ.get('DATABASE_URL') else 'SQLite'
    }
    
    if OCR_AVAILABLE:
        try:
            import pytesseract
            # Test Tesseract
            if tesseract_path and os.path.exists(tesseract_path):
                result = subprocess.run([tesseract_path, '--version'], 
                                      capture_output=True, text=True)
                info['tesseract_version'] = result.stdout[:100] if result.stdout else None
            
            # Test pytesseract
            test_img = Image.new('RGB', (100, 50), color='white')
            test_text = pytesseract.image_to_string(test_img)
            info['pytesseract_test'] = f"OK (output length: {len(test_text)})"
            
        except Exception as e:
            info['test_error'] = str(e)
    
    return jsonify(info)

@app.route('/api/receipt/test', methods=['GET'])
def test_ocr():
    return jsonify({
        'ocr_available': OCR_AVAILABLE,
        'system_ready': True,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'environment': 'Render' if os.environ.get('RENDER') else 'Local'
    })

# ========== FRONTEND TEMPLATE ==========
def get_frontend_template(is_admin=False):
    """Get appropriate frontend template based on user type"""
    if is_admin:
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Receipt Reward System - Admin Panel</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
            color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #121212;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            overflow: hidden;
            border: 1px solid #2a2a2a;
        }
        
        .header {
            background: linear-gradient(135deg, #2a1a1a 0%, #1a0a0a 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            border-bottom: 1px solid #442222;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
            color: #ffcccc;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
            color: #ff9999;
        }
        
        .content {
            padding: 40px;
            background: #0a0a0a;
        }
        
        .card {
            background: #1a1a1a;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #442222;
        }
        
        .card h2 {
            color: #ffcccc;
            margin-bottom: 20px;
            font-size: 1.6rem;
            border-bottom: 3px solid #662222;
            padding-bottom: 12px;
        }
        
        input[type="text"], 
        input[type="number"],
        input[type="password"] {
            width: 100%;
            padding: 16px;
            margin: 12px 0;
            border: 2px solid #552222;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s;
            background: #222;
            color: #ffffff;
        }
        
        input:focus {
            outline: none;
            border-color: #ff6666;
            box-shadow: 0 0 0 4px rgba(255, 102, 102, 0.1);
        }
        
        button {
            background: linear-gradient(135deg, #552222 0%, #662222 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin: 8px 5px;
            border: 1px solid #662222;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(255, 102, 102, 0.2);
            background: linear-gradient(135deg, #662222 0%, #772222 100%);
        }
        
        .status-box {
            background: #222;
            border-left: 4px solid #662222;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1rem;
            line-height: 1.5;
            border: 1px solid #333;
        }
        
        .status-box.success {
            background: #1a2a1a;
            border-left-color: #4CAF50;
        }
        
        .status-box.error {
            background: #2a1a1a;
            border-left-color: #f44336;
        }
        
        .status-box.info {
            background: #1a2a2a;
            border-left-color: #2196F3;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #222;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .data-table th {
            background: #332222;
            color: #ffcccc;
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #442222;
        }
        
        .data-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #333;
        }
        
        .data-table tr:hover {
            background: #2a2a2a;
        }
        
        .admin-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        
        @media (max-width: 768px) {
            .admin-section {
                grid-template-columns: 1fr;
            }
        }
        
        .badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-verified {
            background: #1a3a1a;
            color: #66ff66;
        }
        
        .badge-unverified {
            background: #3a1a1a;
            color: #ff6666;
        }
        
        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Receipt Reward System - Admin Panel</h1>
            <p>Administrator Dashboard - System Management</p>
        </div>
        
        <div class="content">
            <div class="card">
                <h2>📊 System Statistics</h2>
                <button onclick="loadSystemStats()">Refresh Statistics</button>
                <div id="systemStats" class="status-box info">
                    Click "Refresh Statistics" to load system data
                </div>
                
                <div id="statsDetails" class="hidden">
                    <div class="admin-section">
                        <div>
                            <h3>📈 Quick Stats</h3>
                            <div id="quickStats" class="status-box"></div>
                        </div>
                        <div>
                            <h3>🏆 Leaderboard</h3>
                            <div id="leaderboard" class="status-box"></div>
                        </div>
                    </div>
                    
                    <h3>🔄 Recent Activity (Last 24 Hours)</h3>
                    <div id="recentActivity" class="status-box">
                        Loading recent activity...
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>👥 User Management</h2>
                <div class="admin-section">
                    <div>
                        <h3>🔍 Search User</h3>
                        <input type="text" id="searchUserId" placeholder="User ID">
                        <button onclick="searchUser()">Search User</button>
                        <div id="userDetails" class="status-box">
                            Enter User ID to search
                        </div>
                    </div>
                    
                    <div>
                        <h3>📋 All Users</h3>
                        <button onclick="loadAllUsers()">Load All Users</button>
                        <div id="allUsers" class="status-box">
                            Click "Load All Users" to view all registered users
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>⚙️ Admin Tools</h2>
                <div class="admin-section">
                    <div>
                        <h3>📁 System Info</h3>
                        <button onclick="checkSystemHealth()">Check System Health</button>
                        <div id="systemHealth" class="status-box">
                            Click to check system health
                        </div>
                    </div>
                    
                    <div>
                        <h3>🔐 Admin Session</h3>
                        <p style="color: #aaa; margin: 10px 0;">Admin session will expire when you close this tab</p>
                        <button onclick="logoutAdmin()" style="background: #333;">Logout Admin</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Get admin password from URL parameter
        const urlParams = new URLSearchParams(window.location.search);
        const ADMIN_KEY = urlParams.get('password') || 'admin123';
        
        // System Statistics
        async function loadSystemStats() {
            const statsDiv = document.getElementById('systemStats');
            
            try {
                const response = await fetch(`/api/admin/stats?admin_key=${ADMIN_KEY}`);
                const result = await response.json();
                
                if (response.ok) {
                    // Quick stats
                    document.getElementById('quickStats').innerHTML = `
                        Total Users: <strong>${result.total_users}</strong><br>
                        Verified Users: <strong>${result.total_verified_users}</strong><br>
                        Total Receipts: <strong>${result.total_receipts}</strong><br>
                        Total Points Awarded: <strong>${result.total_points_awarded}</strong>
                    `;
                    
                    // Leaderboard
                    let leaderboardHtml = '';
                    if (result.leaderboard && result.leaderboard.length > 0) {
                        result.leaderboard.forEach((user, index) => {
                            leaderboardHtml += `${index + 1}. ${user.username}: ${user.points} points (${user.receipts_count} receipts)<br>`;
                        });
                    } else {
                        leaderboardHtml = 'No users with points yet';
                    }
                    document.getElementById('leaderboard').innerHTML = leaderboardHtml;
                    
                    // Recent activity
                    let activityHtml = '';
                    if (result.recent_activity && result.recent_activity.length > 0) {
                        result.recent_activity.forEach(activity => {
                            const time = new Date(activity.upload_date).toLocaleTimeString();
                            activityHtml += `
                                <div style="margin: 10px 0; padding: 10px; background: #2a2a2a; border-radius: 5px;">
                                    <strong>${activity.username}</strong> uploaded "${activity.filename}"<br>
                                    Amount: RM ${activity.amount || '0.00'} | Points: ${activity.points || 0}<br>
                                    Status: ${activity.status} | Time: ${time}
                                </div>
                            `;
                        });
                    } else {
                        activityHtml = 'No recent activity in the last 24 hours';
                    }
                    document.getElementById('recentActivity').innerHTML = activityHtml;
                    
                    // Show details
                    document.getElementById('statsDetails').classList.remove('hidden');
                    showStatus(statsDiv, '✅ Statistics loaded successfully', 'success');
                } else {
                    showStatus(statsDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(statsDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        // Search User
        async function searchUser() {
            const userId = document.getElementById('searchUserId').value;
            const detailsDiv = document.getElementById('userDetails');
            
            if (!userId) {
                showStatus(detailsDiv, 'Please enter User ID', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/admin/user/${userId}/details?admin_key=${ADMIN_KEY}`);
                const result = await response.json();
                
                if (response.ok) {
                    let html = `
                        <h3>User Details</h3>
                        <p><strong>ID:</strong> ${result.user.id}</p>
                        <p><strong>Username:</strong> ${result.user.username}</p>
                        <p><strong>Telegram ID:</strong> ${result.user.telegram_id || 'Not set'}</p>
                        <p><strong>Email:</strong> ${result.user.email}</p>
                        <p><strong>Points:</strong> ${result.user.points}</p>
                        <p><strong>Status:</strong> <span class="badge ${result.user.is_verified ? 'badge-verified' : 'badge-unverified'}">
                            ${result.user.is_verified ? 'Verified' : 'Not Verified'}
                        </span></p>
                        <p><strong>Joined:</strong> ${new Date(result.user.created_at).toLocaleDateString()}</p>
                        <p><strong>Total Receipts:</strong> ${result.total_receipts}</p>
                        <p><strong>Total Points Earned:</strong> ${result.total_points_earned}</p>
                        
                        <h4 style="margin-top: 20px;">Recent Receipts</h4>
                    `;
                    
                    if (result.receipts && result.receipts.length > 0) {
                        html += '<table class="data-table">';
                        html += '<tr><th>Receipt ID</th><th>Filename</th><th>Amount</th><th>Points</th><th>Status</th><th>Date</th></tr>';
                        
                        result.receipts.slice(0, 10).forEach(receipt => {
                            html += `
                                <tr>
                                    <td>${receipt.id.substring(0, 8)}...</td>
                                    <td>${receipt.filename}</td>
                                    <td>${receipt.amount ? 'RM ' + receipt.amount.toFixed(2) : 'N/A'}</td>
                                    <td>${receipt.points || 0}</td>
                                    <td>${receipt.status}</td>
                                    <td>${new Date(receipt.upload_date).toLocaleDateString()}</td>
                                </tr>
                            `;
                        });
                        
                        html += '</table>';
                        if (result.receipts.length > 10) {
                            html += `<p>... and ${result.receipts.length - 10} more receipts</p>`;
                        }
                    } else {
                        html += '<p>No receipts found for this user</p>';
                    }
                    
                    detailsDiv.innerHTML = html;
                } else {
                    showStatus(detailsDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(detailsDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        // Load All Users
        async function loadAllUsers() {
            const usersDiv = document.getElementById('allUsers');
            
            try {
                const response = await fetch(`/api/admin/users?admin_key=${ADMIN_KEY}`);
                const result = await response.json();
                
                if (response.ok) {
                    let html = `
                        <h3>All Users (${result.total_users})</h3>
                        <table class="data-table">
                            <tr>
                                <th>ID</th>
                                <th>Username</th>
                                <th>Telegram ID</th>
                                <th>Points</th>
                                <th>Status</th>
                                <th>Receipts</th>
                                <th>Joined</th>
                            </tr>
                    `;
                    
                    result.users.forEach(user => {
                        html += `
                            <tr>
                                <td>${user.id}</td>
                                <td>${user.username}</td>
                                <td>${user.telegram_id || 'N/A'}</td>
                                <td>${user.points}</td>
                                <td>
                                    <span class="badge ${user.is_verified ? 'badge-verified' : 'badge-unverified'}">
                                        ${user.is_verified ? 'Verified' : 'Unverified'}
                                    </span>
                                </td>
                                <td>${user.receipts_count}</td>
                                <td>${new Date(user.created_at).toLocaleDateString()}</td>
                            </tr>
                        `;
                    });
                    
                    html += '</table>';
                    usersDiv.innerHTML = html;
                } else {
                    showStatus(usersDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(usersDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        // System Health Check
        async function checkSystemHealth() {
            const healthDiv = document.getElementById('systemHealth');
            
            try {
                // Check multiple endpoints
                const [statsRes, usersRes, ocrRes] = await Promise.all([
                    fetch(`/api/admin/stats?admin_key=${ADMIN_KEY}`),
                    fetch(`/api/admin/users?admin_key=${ADMIN_KEY}`),
                    fetch('/api/debug/ocr')
                ]);
                
                const stats = await statsRes.json();
                const users = await usersRes.json();
                const ocr = await ocrRes.json();
                
                let html = '<h3>System Health Status</h3>';
                html += `<p>✅ Database: Connected (${stats.total_users || 0} users, ${stats.total_receipts || 0} receipts)</p>`;
                html += `<p>✅ OCR System: ${ocr.ocr_available ? 'Available' : 'Not Available'}</p>`;
                html += `<p>✅ Upload Folder: ${ocr.upload_folder_exists ? 'Exists' : 'Missing'}</p>`;
                html += `<p>✅ API Endpoints: All endpoints responding</p>`;
                
                // Check for issues
                const issues = [];
                if (!ocr.ocr_available) {
                    issues.push('OCR is not configured properly');
                }
                if (stats.total_points_awarded === 0 && stats.total_receipts > 0) {
                    issues.push('No points have been awarded despite having receipts');
                }
                
                if (issues.length > 0) {
                    html += '<h4 style="color: #ff6666; margin-top: 15px;">⚠️ Issues Found:</h4>';
                    issues.forEach(issue => {
                        html += `<p>• ${issue}</p>`;
                    });
                } else {
                    html += '<p style="color: #66ff66; margin-top: 15px;">✅ All systems operational</p>';
                }
                
                healthDiv.innerHTML = html;
            } catch (error) {
                showStatus(healthDiv, `Error checking system health: ${error.message}`, 'error');
            }
        }
        
        // Logout Admin
        function logoutAdmin() {
            if (confirm('Are you sure you want to logout from admin panel?')) {
                // Remove admin parameter from URL
                const url = new URL(window.location.href);
                url.searchParams.delete('admin');
                url.searchParams.delete('password');
                window.location.href = url.toString();
            }
        }
        
        // Helper function
        function showStatus(element, message, type) {
            if (element) {
                element.innerHTML = message;
                element.className = 'status-box ' + (type || '');
                element.style.display = 'block';
            }
        }
        
        // Auto-load statistics on page load
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(loadSystemStats, 500);
        });
    </script>
</body>
</html>
        '''
    else:
        # Regular user template (same as your original)
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Receipt Reward System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
            color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #121212;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            overflow: hidden;
            border: 1px solid #2a2a2a;
        }
        
        .header {
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            border-bottom: 1px solid #333;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                rgba(255, 255, 255, 0.05) 0%, 
                rgba(255, 255, 255, 0.02) 25%, 
                transparent 50%, 
                rgba(0, 0, 0, 0.1) 75%);
        }
        
        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
            color: white;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
            color: #cccccc;
        }
        
        .header p strong {
            color: #ffffff;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
            background: #0a0a0a;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: #1a1a1a;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #333;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
            border-color: #444;
        }
        
        .card h2 {
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 1.6rem;
            border-bottom: 3px solid #666;
            padding-bottom: 12px;
            font-weight: 700;
        }
        
        .upload-area {
            border: 3px dashed #555;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            background: #222;
        }
        
        .upload-area:hover {
            border-color: #777;
            background: #2a2a2a;
            transform: scale(1.01);
        }
        
        .upload-area.dragover {
            border-color: #888;
            background: #333;
        }
        
        .upload-icon {
            font-size: 56px;
            color: #888;
            margin-bottom: 20px;
        }
        
        input[type="text"], 
        input[type="number"],
        input[type="email"] {
            width: 100%;
            padding: 16px;
            margin: 12px 0;
            border: 2px solid #444;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s;
            background: #222;
            color: #ffffff;
        }
        
        input[type="text"]:focus, 
        input[type="number"]:focus,
        input[type="email"]:focus {
            outline: none;
            border-color: #666;
            box-shadow: 0 0 0 4px rgba(100, 100, 100, 0.1);
            background: #2a2a2a;
        }
        
        input::placeholder {
            color: #888;
        }
        
        button {
            background: linear-gradient(135deg, #333 0%, #444 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin: 8px 5px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 1px solid #444;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
            background: linear-gradient(135deg, #444 0%, #555 100%);
            border-color: #666;
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #555 0%, #666 100%);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .btn-secondary:hover {
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
            background: linear-gradient(135deg, #666 0%, #777 100%);
        }
        
        .status-box {
            background: #222;
            border-left: 4px solid #666;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1rem;
            line-height: 1.5;
            border: 1px solid #333;
            display: block;
        }
        
        .status-box.success {
            background: #1a2a1a;
            border-left-color: #4CAF50;
            color: #c8e6c9;
            border-color: #2a3a2a;
        }
        
        .status-box.error {
            background: #2a1a1a;
            border-left-color: #f44336;
            color: #ffcdd2;
            border-color: #3a2a2a;
        }
        
        .status-box.info {
            background: #1a2a2a;
            border-left-color: #2196F3;
            color: #bbdefb;
            border-color: #2a3a3a;
        }
        
        .points-display {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a1a 0%, #222 100%);
            border-radius: 15px;
            margin-top: 25px;
            border: 2px solid #444;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }
        
        .points-display h3 {
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 1.4rem;
            font-weight: 600;
        }
        
        .points-value {
            font-size: 4rem;
            font-weight: bold;
            background: linear-gradient(135deg, #ffffff, #cccccc);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 15px 0;
            text-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
        }
        
        .points-display p {
            color: #aaa;
            font-size: 1.1rem;
            margin-top: 10px;
        }
        
        .receipts-list {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        .receipts-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .receipts-list::-webkit-scrollbar-track {
            background: #222;
            border-radius: 4px;
        }
        
        .receipts-list::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #444, #666);
            border-radius: 4px;
        }
        
        .receipts-list::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #555, #777);
        }
        
        .receipt-item {
            background: #222;
            padding: 18px;
            margin: 12px 0;
            border-radius: 12px;
            border-left: 4px solid #666;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            border: 1px solid #333;
            transition: all 0.3s;
        }
        
        .receipt-item:hover {
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
            border-color: #777;
            background: #2a2a2a;
        }
        
        .receipt-info h4 {
            color: #ffffff;
            margin-bottom: 8px;
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .receipt-info p {
            color: #aaa;
            font-size: 0.9rem;
        }
        
        .receipt-amount {
            font-weight: bold;
            color: #4CAF50;
            font-size: 1.4rem;
            text-shadow: 0 2px 5px rgba(76, 175, 80, 0.2);
        }
        
        .receipt-status {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 8px;
            display: inline-block;
            border: 1px solid;
        }
        
        .status-pending { 
            background: #333322; 
            color: #ffcc00; 
            border-color: #665500;
        }
        .status-processing { 
            background: #222233; 
            color: #6699ff; 
            border-color: #334466;
        }
        .status-processed { 
            background: #223322; 
            color: #66cc66; 
            border-color: #336633;
        }
        .status-failed { 
            background: #332222; 
            color: #ff6666; 
            border-color: #663333;
        }
        .status-verified { 
            background: #332244; 
            color: #aa66ff; 
            border-color: #553377;
        }
        
        .progress-bar {
            height: 12px;
            background: #222;
            border-radius: 6px;
            margin: 25px 0;
            overflow: hidden;
            border: 1px solid #333;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #666, #888, #666);
            background-size: 200% 100%;
            width: 0%;
            transition: width 0.5s ease;
            animation: shimmer 2s infinite linear;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        .tabs {
            display: flex;
            border-bottom: 2px solid #333;
            background: #1a1a1a;
            padding: 0 40px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .tab {
            padding: 20px 32px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            font-weight: 600;
            color: #aaa;
            transition: all 0.3s;
            font-size: 1.1rem;
            position: relative;
        }
        
        .tab:hover {
            color: white;
            background: #222;
        }
        
        .tab.active {
            color: white;
            background: #222;
            border-bottom-color: #666;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .upload-area p {
            color: #ccc;
            margin: 10px 0;
            font-size: 1rem;
        }
        
        .upload-area p strong {
            color: white;
            font-weight: 600;
        }
        
        #fileInfo p {
            color: #aaa;
            margin: 8px 0;
            font-size: 0.95rem;
        }
        
        #fileInfo p strong {
            color: #ddd;
        }
        
        @media (max-width: 480px) {
            .header {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2.2rem;
            }
            
            .header p {
                font-size: 1rem;
            }
            
            .content {
                padding: 25px 20px;
            }
            
            .card {
                padding: 25px;
            }
            
            .tab {
                padding: 16px 20px;
                font-size: 1rem;
            }
            
            .tabs {
                padding: 0 20px;
            }
            
            .points-value {
                font-size: 3rem;
            }
        }
        
        /* Hidden elements */
        .hidden {
            display: none !important;
        }
        
        .welcome-banner {
            background: linear-gradient(135deg, #1a2a2a 0%, #2a3a3a 100%);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #334444;
            text-align: center;
        }
        
        .welcome-banner h3 {
            color: #ffffff;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }
        
        .welcome-banner p {
            color: #cccccc;
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="position: relative; z-index: 2;">
                <h1>📸 Receipt Reward System</h1>
                <p>Upload your shopping receipts and earn points equal to the total amount!</p>
                <p style="margin-top: 15px; font-size: 1.3rem; font-weight: 600;">
                    <strong>🎯 RM 1 = 1 Point</strong>
                </p>
                <div style="margin-top: 25px; display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;">
                    <span style="display: inline-flex; align-items: center; gap: 8px; background: rgba(255, 255, 255, 0.1); padding: 8px 16px; border-radius: 20px;">
                        <span style="color: #4CAF50; font-weight: bold;">✓</span> Automatic OCR
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 8px; background: rgba(255, 255, 255, 0.1); padding: 8px 16px; border-radius: 20px;">
                        <span style="color: #4CAF50; font-weight: bold;">✓</span> Instant Points
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 8px; background: rgba(255, 255, 255, 0.1); padding: 8px 16px; border-radius: 20px;">
                        <span style="color: #4CAF50; font-weight: bold;">✓</span> Secure Upload
                    </span>
                </div>
            </div>
        </div>
        
        <!-- Main User Tabs -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')" id="upload-tab-btn">📤 Upload Receipt</div>
            <div class="tab" onclick="switchTab('register')" id="register-tab-btn">👤 Register/Verify</div>
            <div class="tab" onclick="switchTab('receipts')">🧾 My Receipts</div>
            <div class="tab" onclick="switchTab('myaccount')">👤 My Account</div>
        </div>
        
        <!-- Upload Tab -->
        <div class="content tab-content active" id="upload-tab">
            <!-- Welcome banner for verified users -->
            <div id="verifiedUserWelcome" class="welcome-banner hidden">
                <h3>Welcome back, <span id="welcomeUsername"></span>! 👋</h3>
                <p>Your account is verified and ready. Start uploading receipts to earn points!</p>
            </div>
            
            <div class="card">
                <h2>📤 Upload Receipt</h2>
                <div id="userIdSection">
                    <input type="text" id="user_id" placeholder="Enter your User ID">
                    <button onclick="checkUserStatus()">Check Status</button>
                </div>
                
                <div id="uploadSection" class="hidden">
                    <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                        <div class="upload-icon">📄</div>
                        <p><strong>Click to select or drag & drop receipt</strong></p>
                        <p>Supported: PNG, JPG, JPEG, PDF, GIF, BMP</p>
                        <p>Max size: 16MB</p>
                        <input type="file" id="fileInput" accept="image/*,application/pdf" style="display: none;">
                        <div id="fileInfo"></div>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" id="uploadProgress"></div>
                    </div>
                    
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <button onclick="uploadReceipt()" id="uploadBtn">Upload & Process</button>
                        <button onclick="clearUpload()" class="btn-secondary">Clear</button>
                    </div>
                    
                    <div id="upload_status" class="status-box"></div>
                </div>
                
                <div class="points-display">
                    <h3>Your Current Points</h3>
                    <div class="points-value" id="currentPoints">0</div>
                    <p id="userName">Enter User ID to see points</p>
                </div>
            </div>
        </div>
        
        <!-- Register/Verify Tab -->
        <div class="content tab-content" id="register-tab">
            <div id="unverifiedUserInfo" class="status-box info">
                <h3>🔐 Account Registration & Verification</h3>
                <p>To use the Receipt Reward System, you need to:</p>
                <ol style="margin-left: 20px; margin-top: 10px;">
                    <li>Register with your Telegram ID</li>
                    <li>Check Telegram for verification code</li>
                    <li>Enter the code here to verify your account</li>
                    <li>Start uploading receipts and earning points!</li>
                </ol>
            </div>
            
            <div class="card">
                <h2>👤 Register with Telegram</h2>
                <input type="text" id="register_username" placeholder="Username">
                <input type="text" id="telegram_id" placeholder="Your Telegram ID (e.g., 123456789)">
                <button onclick="registerUser()">Register via Telegram</button>
                <div id="register_status" class="status-box"></div>
            </div>
            
            <div class="card">
                <h2>✅ Verify Your Account</h2>
                <p style="color: #aaa; margin-bottom: 20px;">
                    After registration, check Telegram for verification code
                </p>
                <input type="text" id="verify_user_id" placeholder="Your User ID">
                <input type="text" id="verify_code" placeholder="6-digit Verification Code">
                <button onclick="verifyAccount()">Verify Account</button>
                <div id="verify_status" class="status-box"></div>
            </div>
        </div>
        
        <!-- Receipts Tab -->
        <div class="content tab-content" id="receipts-tab">
            <div class="card">
                <h2>🧾 My Receipts</h2>
                <div id="receiptsUserIdSection">
                    <input type="text" id="receipts_user_id" placeholder="Enter User ID">
                    <button onclick="checkReceiptsUserStatus()">Load Receipts</button>
                </div>
                
                <div id="receiptsContent" class="hidden">
                    <div class="receipts-list" id="receiptsList">
                        <!-- Receipts will be loaded here -->
                    </div>
                    
                    <div id="receipts_status" class="status-box"></div>
                </div>
            </div>
        </div>
        
        <!-- My Account Tab -->
        <div class="content tab-content" id="myaccount-tab">
            <div class="card">
                <h2>👤 My Account Information</h2>
                <div id="myAccountInfo" class="status-box info">
                    <p>Enter your User ID to view your account information.</p>
                </div>
                <input type="text" id="myaccount_user_id" placeholder="Enter your User ID">
                <button onclick="loadMyAccount()">Load My Account</button>
                
                <div id="accountDetails" class="hidden">
                    <div class="status-box success" style="margin-bottom: 20px;">
                        <h3>Account Details</h3>
                        <p><strong>Username:</strong> <span id="accountUsername"></span></p>
                        <p><strong>User ID:</strong> <span id="accountUserId"></span></p>
                        <p><strong>Telegram ID:</strong> <span id="accountTelegramId"></span></p>
                        <p><strong>Status:</strong> <span id="accountStatus"></span></p>
                        <p><strong>Joined:</strong> <span id="accountJoined"></span></p>
                    </div>
                    
                    <div class="status-box info">
                        <h3>Quick Actions</h3>
                        <button onclick="checkMyPoints()" style="margin: 5px;">Check My Points</button>
                        <button onclick="loadMyReceiptsFromAccount()" style="margin: 5px;">View My Receipts</button>
                        <button onclick="switchTab('upload')" style="margin: 5px;">Upload New Receipt</button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>✏️ Manual Receipt Entry</h2>
                <p style="color: #aaa; margin-bottom: 20px;">
                    If OCR failed to process your receipt, you can enter the amount manually.
                    Note: This will be verified by the system.
                </p>
                <input type="text" id="myaccount_receipt_id" placeholder="Your Receipt ID">
                <input type="number" id="myaccount_amount" placeholder="Amount (RM)" step="0.01" min="0.01">
                <button onclick="manualAmountEntryMyAccount()">Submit Amount</button>
                <div id="myaccount_manual_status" class="status-box"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let currentUserId = null;
        let currentUsername = null;
        let isUserVerified = false;
        
        // Tab switching - FIXED VERSION (no event parameter issues)
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            const tabElement = document.getElementById(tabName + '-tab');
            if (tabElement) {
                tabElement.classList.add('active');
            }
            
            // Activate the corresponding tab button
            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(btn => {
                if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabName)) {
                    btn.classList.add('active');
                }
            });
            
            // Special handling for verified users
            if (isUserVerified && tabName === 'upload') {
                // Auto-fill user ID if available
                if (currentUserId) {
                    document.getElementById('user_id').value = currentUserId;
                    showUploadSection();
                }
            }
            
            // Clear any error messages when switching tabs
            if (tabName === 'upload') {
                document.getElementById('upload_status').innerHTML = '';
            }
        }
        
        // Check user status
        async function checkUserStatus() {
            const userId = document.getElementById('user_id').value;
            const statusDiv = document.getElementById('upload_status');
            
            if (!userId) {
                showStatus(statusDiv, 'Please enter your User ID', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/user/${userId}/status`);
                const result = await response.json();
                
                if (response.ok) {
                    currentUserId = userId;
                    currentUsername = result.username;
                    isUserVerified = result.is_verified;
                    
                    // Save to localStorage
                    localStorage.setItem('receiptRewardUserId', userId);
                    localStorage.setItem('receiptRewardUsername', result.username);
                    localStorage.setItem('receiptRewardVerified', result.is_verified);
                    
                    if (result.is_verified) {
                        // User is verified - show upload section
                        showStatus(statusDiv, 
                            `✅ Account verified! Welcome back, ${result.username}!<br>
                            You have ${result.points} points from ${result.receipts_count} receipts.`, 
                            'success'
                        );
                        
                        showUploadSection();
                        
                        // Show welcome banner
                        document.getElementById('welcomeUsername').textContent = result.username;
                        document.getElementById('verifiedUserWelcome').classList.remove('hidden');
                        
                        // Update points display
                        document.getElementById('currentPoints').textContent = result.points;
                        document.getElementById('userName').textContent = `User: ${result.username}`;
                        
                        // Switch to upload tab if not already
                        switchTab('upload');
                        
                        // Hide register/verify tab from navigation
                        document.getElementById('register-tab-btn').classList.add('hidden');
                    } else {
                        // User not verified - show verification prompt
                        showStatus(statusDiv, 
                            `⚠️ Account not verified.<br>
                            Please complete Telegram verification in the Register/Verify tab.`, 
                            'error'
                        );
                        
                        // Switch to register tab
                        switchTab('register');
                        document.getElementById('verify_user_id').value = userId;
                    }
                } else {
                    showStatus(statusDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(statusDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        function showUploadSection() {
            const userIdSection = document.getElementById('userIdSection');
            const uploadSection = document.getElementById('uploadSection');
            
            if (userIdSection) userIdSection.classList.add('hidden');
            if (uploadSection) uploadSection.classList.remove('hidden');
        }
        
        // Registration function
        async function registerUser() {
            const username = document.getElementById('register_username').value;
            const telegramId = document.getElementById('telegram_id').value;
            const statusDiv = document.getElementById('register_status');
    
            if (!username || !telegramId) {
                showStatus(statusDiv, 'Please fill in all fields', 'error');
                return;
            }
    
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, telegram_id: telegramId})
            });
    
            const result = await response.json();
    
            if (response.ok) {
                showStatus(statusDiv, 
                    `✅ Registration successful!<br>
                    User ID: <strong>${result.user_id}</strong><br>
                    Check Telegram for verification code (6 digits)<br>
                    Enter the code in the verification section below.`, 
                    'success'
                );
                // Auto-fill verification form
                document.getElementById('verify_user_id').value = result.user_id;
            } else {
                showStatus(statusDiv, `❌ Error: ${result.error}`, 'error');
            }
        }
        
        // Verification function
        async function verifyAccount() {
            const userId = document.getElementById('verify_user_id').value;
            const code = document.getElementById('verify_code').value;
            const statusDiv = document.getElementById('verify_status');
    
            if (!userId || !code) {
                showStatus(statusDiv, 'Please fill in all fields', 'error');
                return;
            }
    
            try {
                const response = await fetch('/api/verify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        verification_code: code
                    })
                });
        
                const result = await response.json();
        
                if (response.ok) {
                    showStatus(statusDiv, 
                        `✅ Account verified successfully!<br>
                        You can now upload receipts and earn points.<br>
                        Go to the Upload Receipt tab to get started.`, 
                        'success'
                    );
            
                    // Auto-fill user_id in upload section
                    document.getElementById('user_id').value = userId;
                    
                    // Set global variables
                    currentUserId = userId;
                    currentUsername = result.username;
                    isUserVerified = true;
                    
                    // Save to localStorage
                    localStorage.setItem('receiptRewardUserId', userId);
                    localStorage.setItem('receiptRewardUsername', result.username);
                    localStorage.setItem('receiptRewardVerified', true);
                    
                    // Auto-switch to upload tab after 2 seconds
                    setTimeout(() => {
                        switchTab('upload');
                        checkUserStatus(); // This will now show the upload section
                    }, 2000);
                } else {
                    showStatus(statusDiv, `❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(statusDiv, `❌ Error: ${error.message}`, 'error');
            }
        }
        
        // Check receipts user status
        async function checkReceiptsUserStatus() {
            const userId = document.getElementById('receipts_user_id').value;
            const statusDiv = document.getElementById('receipts_status');
            
            if (!userId) {
                showStatus(statusDiv, 'Please enter User ID', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/user/${userId}/status`);
                const result = await response.json();
                
                if (response.ok && result.is_verified) {
                    currentUserId = userId;
                    currentUsername = result.username;
                    isUserVerified = true;
                    
                    document.getElementById('receiptsUserIdSection').classList.add('hidden');
                    document.getElementById('receiptsContent').classList.remove('hidden');
                    loadUserReceipts();
                } else {
                    showStatus(statusDiv, 
                        'User not found or not verified. Please check your User ID.', 
                        'error'
                    );
                }
            } catch (error) {
                showStatus(statusDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        // My Account functions
        async function loadMyAccount() {
            const userId = document.getElementById('myaccount_user_id').value;
            const infoDiv = document.getElementById('myAccountInfo');
            
            if (!userId) {
                showStatus(infoDiv, 'Please enter your User ID', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/user/${userId}/status`);
                const result = await response.json();
                
                if (response.ok) {
                    currentUserId = userId;
                    currentUsername = result.username;
                    isUserVerified = result.is_verified;
                    
                    // Update account details
                    document.getElementById('accountUsername').textContent = result.username;
                    document.getElementById('accountUserId').textContent = result.user_id;
                    document.getElementById('accountTelegramId').textContent = result.telegram_id || 'Not set';
                    document.getElementById('accountStatus').textContent = result.is_verified ? '✅ Verified' : '❌ Not Verified';
                    document.getElementById('accountJoined').textContent = new Date(result.created_at).toLocaleDateString();
                    
                    // Show account details
                    document.getElementById('accountDetails').classList.remove('hidden');
                    
                    // Update welcome message if this is the current user
                    if (userId === localStorage.getItem('receiptRewardUserId')) {
                        document.getElementById('welcomeUsername').textContent = result.username;
                        document.getElementById('verifiedUserWelcome').classList.remove('hidden');
                    }
                    
                } else {
                    showStatus(infoDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(infoDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        async function checkMyPoints() {
            if (!currentUserId) {
                alert('Please load your account first');
                return;
            }
            
            try {
                const response = await fetch(`/api/user/${currentUserId}/points`);
                const result = await response.json();
                
                if (response.ok) {
                    showStatus(document.getElementById('myAccountInfo'), 
                        `Your Points: <strong>${result.points}</strong><br>
                        Total Receipts: <strong>${result.receipts_count}</strong>`, 
                        'success'
                    );
                }
            } catch (error) {
                console.error('Error checking points:', error);
            }
        }
        
        async function loadMyReceiptsFromAccount() {
            if (!currentUserId) {
                alert('Please load your account first');
                return;
            }
            
            // Switch to receipts tab and load receipts
            switchTab('receipts');
            document.getElementById('receipts_user_id').value = currentUserId;
            setTimeout(() => {
                checkReceiptsUserStatus();
            }, 100);
        }
        
        async function manualAmountEntryMyAccount() {
            const receiptId = document.getElementById('myaccount_receipt_id').value;
            const amount = document.getElementById('myaccount_amount').value;
            const statusDiv = document.getElementById('myaccount_manual_status');
            
            if (!receiptId || !amount) {
                showStatus(statusDiv, 'Please fill in all fields', 'error');
                return;
            }
            
            try {
                const response = await fetch(`/api/receipt/${receiptId}/manual_amount`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({amount: parseFloat(amount)})
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus(statusDiv, 
                        `✓ Receipt manually verified!<br>
                        Points awarded: <strong>${result.points_awarded}</strong><br>
                        New total points: <strong>${result.new_user_points}</strong>`, 
                        'success'
                    );
                } else {
                    showStatus(statusDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(statusDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        // Drag and drop functionality
        const dropArea = document.getElementById('dropArea');
        const fileInput = document.getElementById('fileInput');
        
        if (dropArea && fileInput) {
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
            });
            
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const files = e.dataTransfer.files;
                handleFiles(files);
            }
            
            fileInput.addEventListener('change', function() {
                handleFiles(this.files);
            });
        }
        
        function handleFiles(files) {
            if (files.length > 0 && document.getElementById('fileInfo')) {
                const file = files[0];
                document.getElementById('fileInfo').innerHTML = `
                    <p><strong>Selected file:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                `;
            }
        }
        
        async function uploadReceipt() {
            if (!currentUserId || !isUserVerified) {
                showStatus(document.getElementById('upload_status'), 'Please verify your account first', 'error');
                return;
            }
            
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('upload_status');
            const progressBar = document.getElementById('uploadProgress');
            const uploadBtn = document.getElementById('uploadBtn');
            
            if (!fileInput || !fileInput.files[0]) {
                showStatus(statusDiv, 'Please select a receipt file', 'error');
                return;
            }
            
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = 'Uploading...';
            if (progressBar) progressBar.style.width = '30%';
            
            const formData = new FormData();
            formData.append('user_id', currentUserId);
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (progressBar) progressBar.style.width = '70%';
                const result = await response.json();
                
                if (response.ok) {
                    showStatus(statusDiv, 
                        `✓ Receipt uploaded successfully!<br>
                        Receipt ID: <strong>${result.receipt_id}</strong><br>
                        Status: Processing...`, 
                        'success'
                    );
                    
                    if (progressBar) progressBar.style.width = '100%';
                    pollReceiptStatus(result.receipt_id);
                    
                    // Update points display after 2 seconds
                    setTimeout(() => {
                        fetchUserPoints(currentUserId);
                    }, 2000);
                } else {
                    showStatus(statusDiv, `✗ Error: ${result.error}`, 'error');
                    if (progressBar) progressBar.style.width = '0%';
                }
            } catch (error) {
                showStatus(statusDiv, `✗ Upload failed: ${error.message}`, 'error');
                if (progressBar) progressBar.style.width = '0%';
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = 'Upload & Process';
            }
        }
        
        async function fetchUserPoints(userId) {
            try {
                const response = await fetch(`/api/user/${userId}/points`);
                const result = await response.json();
                
                if (response.ok && document.getElementById('currentPoints')) {
                    document.getElementById('currentPoints').textContent = result.points;
                }
            } catch (error) {
                console.error('Points update error:', error);
            }
        }
        
        async function pollReceiptStatus(receiptId) {
            const statusDiv = document.getElementById('upload_status');
            
            try {
                const response = await fetch(`/api/receipt/${receiptId}`);
                const result = await response.json();
                
                if (result.status === 'processed' || result.status === 'verified') {
                    showStatus(statusDiv, 
                        `🎉 Receipt processed successfully!<br>
                        Amount: <strong>RM ${result.extracted_amount?.toFixed(2) || '0.00'}</strong><br>
                        Points awarded: <strong>${result.points_awarded || 0}</strong>`, 
                        'success'
                    );
                    fetchUserPoints(currentUserId);
                } else if (result.status === 'failed' || result.status === 'error') {
                    showStatus(statusDiv, 
                        `⚠️ OCR processing failed.<br>
                        Receipt ID: <strong>${receiptId}</strong><br>
                        Please use manual entry in My Account tab.`, 
                        'error'
                    );
                } else if (result.status === 'processing') {
                    setTimeout(() => pollReceiptStatus(receiptId), 2000);
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
        
        async function loadUserReceipts() {
            if (!currentUserId) return;
            
            const receiptsList = document.getElementById('receiptsList');
            const statusDiv = document.getElementById('receipts_status');
            
            try {
                const response = await fetch(`/api/user/${currentUserId}/receipts`);
                const result = await response.json();
                
                if (response.ok) {
                    if (result.receipts.length === 0) {
                        receiptsList.innerHTML = '<p style="color: #888; text-align: center; padding: 40px;">No receipts found. Start by uploading your first receipt!</p>';
                    } else {
                        let html = '';
                        result.receipts.forEach(receipt => {
                            const date = new Date(receipt.upload_date);
                            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                            
                            html += `
                                <div class="receipt-item">
                                    <div class="receipt-info">
                                        <h4>${receipt.filename}</h4>
                                        <p>${formattedDate}</p>
                                        <span class="receipt-status status-${receipt.status}">${receipt.status}</span>
                                    </div>
                                    <div>
                                        ${receipt.amount ? `<div class="receipt-amount">RM ${receipt.amount.toFixed(2)}</div>` : ''}
                                        ${receipt.points ? `<div style="color: #888; font-weight: 600;">Points: ${receipt.points}</div>` : ''}
                                    </div>
                                </div>
                            `;
                        });
                        
                        receiptsList.innerHTML = html;
                    }
                    
                    showStatus(statusDiv, 
                        `Loaded ${result.total_receipts} receipts for ${result.username}<br>
                        Total Points: <strong>${result.total_points}</strong>`, 
                        'success'
                    );
                } else {
                    showStatus(statusDiv, `Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(statusDiv, `Error: ${error.message}`, 'error');
            }
        }
        
        function clearUpload() {
            if (document.getElementById('fileInput')) document.getElementById('fileInput').value = '';
            if (document.getElementById('fileInfo')) document.getElementById('fileInfo').innerHTML = '';
            if (document.getElementById('upload_status')) document.getElementById('upload_status').innerHTML = '';
            const progressBar = document.getElementById('uploadProgress');
            if (progressBar) progressBar.style.width = '0%';
        }
        
        function showStatus(element, message, type) {
            if (element) {
                element.innerHTML = message;
                element.className = 'status-box ' + (type || '');
                element.style.display = 'block';
            }
        }
        
        // Check OCR status on load
        document.addEventListener('DOMContentLoaded', function() {
            fetch('/api/debug/ocr')
                .then(response => response.json())
                .then(data => {
                    if (!data.ocr_available) {
                        const statusDiv = document.getElementById('upload_status');
                        showStatus(statusDiv, 
                            '⚠️ OCR is not fully configured. You can still upload receipts and use manual amount entry.<br>' +
                            'For automatic OCR processing, ensure Tesseract OCR is installed.', 
                            'error'
                        );
                    }
                });
            
            // Check if there's a user ID in localStorage (for returning users)
            const savedUserId = localStorage.getItem('receiptRewardUserId');
            const savedUsername = localStorage.getItem('receiptRewardUsername');
            const savedVerified = localStorage.getItem('receiptRewardVerified');
            
            if (savedUserId) {
                document.getElementById('user_id').value = savedUserId;
                currentUserId = savedUserId;
                
                if (savedUsername) {
                    currentUsername = savedUsername;
                }
                
                if (savedVerified === 'true') {
                    isUserVerified = true;
                    // Auto-check status after a short delay
                    setTimeout(() => {
                        checkUserStatus();
                    }, 500);
                }
            }
        });
    </script>
</body>
</html>
        '''

@app.route('/')
def index():
    """Main route - checks for admin parameter with password"""
    # Check if admin parameter is present with correct password
    is_admin = False
    admin_param = request.args.get('admin')
    admin_pass = request.args.get('password')
    
    if admin_param == '1' and admin_pass == ADMIN_PASSWORD:
        is_admin = True
        print(f"Admin access granted from IP: {request.remote_addr}")
    
    return render_template_string(get_frontend_template(is_admin))

# ========== MAIN ==========
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("=" * 60)
        print("Receipt Reward System - DUAL ENVIRONMENT")
        print("=" * 60)
        print(f"Environment: {'Render' if os.environ.get('RENDER') else 'Local'}")
        print(f"Database: {'PostgreSQL' if os.environ.get('DATABASE_URL') else 'SQLite'}")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"OCR Available: {OCR_AVAILABLE}")
        print(f"Tesseract Path: {tesseract_path}")
        print("=" * 60)
        print(f"User Interface: http://127.0.0.1:5000")
        print(f"Admin Interface: http://127.0.0.1:5000/?admin=1&password={ADMIN_PASSWORD}")
        print("=" * 60)
        print("ADMIN ACCESS:")
        print(f"• Admin password: {ADMIN_PASSWORD}")
        print("• Change ADMIN_PASSWORD in .env file for production")
        print("=" * 60)
        print("ENVIRONMENT VARIABLES REQUIRED FOR RENDER:")
        print("TELEGRAM_BOT_TOKEN=your_bot_token")
        print("TELEGRAM_BOT_USERNAME=your_bot_username")
        print("SECRET_KEY=your_secret_key")
        print("ADMIN_PASSWORD=your_admin_password")
        print("DATABASE_URL=auto-provided-by-render")
        print("=" * 60)
        print("USER FEATURES:")
        print("• Verified users see upload section immediately")
        print("• Telegram verification required for new users")
        print("• Duplicate receipt detection with image/content hashing")
        print("• Real-time Telegram notifications")
        print("• Points system: RM 1 = 1 Point")
        print("=" * 60)
    
    # Get port from environment (Render sets PORT)
    port = int(os.environ.get('PORT', 5000))
    
    # Debug mode only for local development
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)