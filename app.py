from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sqlite3
import numpy as np
import joblib
import json
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

import smtplib
from email.message import EmailMessage
from datetime import datetime

# Email configuration (move to config file in production)
EMAIL_CONFIG = {
    'from_email': '1ds22is002@dsce.edu.in',
    'from_password': 'wsma kcro yhfk cqqf',  # App password
    'alert_recipients': [
        '1ds22is053@dsce.edu.in',
        '1ds22is016@dsce.edu.in',
        '1ds22is181@dsce.edu.in'
    ],
    'admin_recipient': 'admin@healthdept.gov.in'  # Add admin email
}

def normalize_recipients(value):
    """Accept a single email or list and return a clean recipient list."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [email.strip() for email in value if isinstance(email, str) and email.strip()]

def send_email(to_email_addr, subject, body):
    """Send email using configured credentials"""
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['From'] = EMAIL_CONFIG['from_email']
        msg['To'] = to_email_addr
        msg['Subject'] = subject

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_CONFIG['from_email'], EMAIL_CONFIG['from_password'])
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully to {to_email_addr}")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def send_email_to_many(to_emails, subject, body):
    """Send same email to multiple recipients."""
    for email in normalize_recipients(to_emails):
        send_email(email, subject, body)

def hash_password(password):
    """Hash a plain-text password before storing it."""
    return generate_password_hash(password)

def verify_password(stored_password, provided_password):
    """Verify a password and support legacy plain-text records."""
    if stored_password is None:
        return False
    if stored_password.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_password, provided_password)
    return stored_password == provided_password

def send_risk_alert(asha_name, location, risk_level, prediction_data):
    """Send risk alert email based on prediction"""
    
    # Prepare email content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    subject = f" HIGH RISK ALERT - Water-Borne Disease Outbreak Risk in {location}"
    
    body = f"""
     HIGH RISK ALERT - IMMEDIATE ACTION REQUIRED 
    
    ================================================
    WATER-BORNE DISEASE SURVEILLANCE SYSTEM
    ================================================
    
    A HIGH RISK prediction has been recorded in your area!
    
     Location Details:
    -------------------
    Area: {location}
    Reported by: {asha_name} (ASHA Worker)
    Time: {timestamp}
    
     Water Quality Parameters:
    ---------------------------
    • Temperature: {prediction_data.get('temperature', 'N/A')}°C
    • pH Level: {prediction_data.get('ph', 'N/A')}
    • Turbidity: {prediction_data.get('turbidity', 'N/A')} NTU
    • Bacterial Count: {prediction_data.get('bacterial_count', 'N/A')} CFU/mL
    • Dissolved Oxygen: {prediction_data.get('dissolved_oxygen', 'N/A')} mg/L
    • Nitrate Level: {prediction_data.get('nitrate', 'N/A')} mg/L
    • Lead Level: {prediction_data.get('lead', 'N/A')} mg/L
    
     Infrastructure Status:
    ------------------------
    • Clean Water Availability: {prediction_data.get('clean_water_percentage', 'N/A')}%
    • Sanitation Level: {prediction_data.get('sanitation_level', 'N/A')}%
    • Healthcare Access: {prediction_data.get('healthcare_access', 'N/A')}%
    
     Reported Symptoms:
    -------------------
    • Diarrhea: {'Yes' if prediction_data.get('symptom_diarrhea') else 'No'}
    • Fever: {'Yes' if prediction_data.get('symptom_fever') else 'No'}
    
     Water Issues:
    ---------------
    • Dirty Water: {'Yes' if prediction_data.get('water_dirty') else 'No'}
    • Water Scarcity: {'Yes' if prediction_data.get('water_scarcity') else 'No'}
    
     Environmental Factors:
    ------------------------
    • Rainfall: {prediction_data.get('rainfall', 'N/A')} mm
    
     RISK ASSESSMENT: {risk_level.upper()}
    
     RECOMMENDED ACTIONS:
    -----------------------
    1.  IMMEDIATE: Alert local health authorities
    2.  Deploy medical team for health screening
    3.  Distribute chlorine tablets/water purifiers
    4.  Issue public health advisory in affected area
    5.  Conduct additional water quality tests
    6.  Set up temporary health camps
    7.  Launch awareness campaign on hygiene
    
     Response Timeline: Action required within 24 hours
    
    This is an automated alert from Water-Borne Disease Surveillance System.
    Please take immediate action to prevent potential outbreak.
    
    ---
    System Administrator: Contact for any issues
    """
    
    # Send to alert recipients (supports multiple)
    recipients = EMAIL_CONFIG.get('alert_recipients', EMAIL_CONFIG.get('alert_recipient'))
    send_email_to_many(recipients, subject, body)
    
    # Also send to admin email
    if EMAIL_CONFIG.get('admin_recipient'):
        send_email(EMAIL_CONFIG['admin_recipient'], f"[ADMIN] {subject}", body)
    
    # Optional: Send SMS notification (if SMS gateway configured)
    # send_sms_alert(phone_number, f"HIGH RISK Alert in {location}")

def send_medium_risk_alert(asha_name, location, risk_level, prediction_data):
    """Send medium risk notification"""
    subject = f" MEDIUM RISK Alert - Monitor Situation in {location}"
    
    body = f"""
    MEDIUM RISK ALERT - Monitor Required
    ===================================
    
    Location: {location}
    ASHA Worker: {asha_name}
    Risk Level: {risk_level.upper()}
    Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Key Parameters:
    - Bacterial Count: {prediction_data.get('bacterial_count', 'N/A')} CFU/mL
    - Clean Water: {prediction_data.get('clean_water_percentage', 'N/A')}%
    - Reported Symptoms: {'Yes' if prediction_data.get('symptom_diarrhea') or prediction_data.get('symptom_fever') else 'No'}
    
    Recommended Actions:
    1. Monitor water quality regularly
    2. Conduct community awareness session
    3. Prepare for potential escalation
    4. Increase surveillance in the area
    
    This is a preventive alert. Maintain vigilance.
    """
    
    recipients = EMAIL_CONFIG.get('alert_recipients', EMAIL_CONFIG.get('alert_recipient'))
    send_email_to_many(recipients, subject, body)

def send_low_risk_update(asha_name, location, prediction_data):
    """Send low risk update (optional - can be disabled to avoid spam)"""
    # Uncomment if you want low risk notifications
    # subject = f"✅ LOW RISK Update - {location}"
    # body = f"Low risk status maintained in {location}. Continue standard monitoring."
    # send_email(EMAIL_CONFIG['alert_recipient'], subject, body)
    pass


app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'
CORS(app)

# Load ML Model
try:
    model = joblib.load("ml_model/svm_model.pkl")
    scaler = joblib.load("ml_model/scaler.pkl")
    print("ML Model loaded successfully")
except Exception as e:
    print(f"Error loading ML model: {e}")
    model = None
    scaler = None

# Database Setup
def init_db():
    conn = sqlite3.connect('database/water_disease.db')
    cursor = conn.cursor()
    
    # Locals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            area TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # ASHA workers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS asha_workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            worker_id TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            area TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Complaints table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id INTEGER,
            local_name TEXT,
            location TEXT NOT NULL,
            sub_location TEXT NOT NULL,
            symptoms TEXT,
            water_issues TEXT,
            description TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (local_id) REFERENCES locals(id)
        )
    ''')
    
    # Predictions table - FIXED COLUMN ORDER
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asha_id INTEGER,
            asha_name TEXT,
            temperature REAL,
            rainfall REAL,
            ph REAL,
            turbidity REAL,
            dissolved_oxygen REAL,
            nitrate REAL,
            lead REAL,
            bacterial_count REAL,
            clean_water_percentage REAL,
            sanitation_level REAL,
            healthcare_access REAL,
            symptom_diarrhea INTEGER,
            symptom_fever INTEGER,
            water_dirty INTEGER,
            water_scarcity INTEGER,
            prediction TEXT,
            location TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    
    # Sessions table for tracking logins
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_type TEXT,
            user_id INTEGER,
            user_name TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_time TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
if not os.path.exists('database'):
    os.makedirs('database')
init_db()

# ================================
# ROUTES - PAGE RENDERING
# ================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/local/login')
def local_login():
    return render_template('local_login.html')

@app.route('/local/register')
def local_register():
    return render_template('local_register.html')

@app.route('/local/logged')
def local_logged():
    return render_template('local_logged.html')

@app.route('/asha/login')
def asha_login():
    return render_template('asha_login.html')

@app.route('/asha/register')
def asha_register():
    return render_template('asha_register.html')

@app.route('/asha/dashboard')
def asha_dashboard():
    return render_template('asha_logged.html')

@app.route('/asha/predict')
def asha_predict():
    return render_template('asha_predict.html')

@app.route('/admin/login')
def admin_login():
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/control')
def admin_control():
    return render_template('admin_control.html')

@app.route('/logout')
def logout():
    if 'user_id' in session or 'asha_id' in session:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        user_type = session.get('user_type')
        user_id = session.get('user_id') or session.get('asha_id')
        
        cursor.execute('''
            UPDATE user_sessions 
            SET logout_time = ? 
            WHERE user_type = ? AND user_id = ? AND logout_time IS NULL
        ''', (datetime.now(), user_type, user_id))
        
        conn.commit()
        conn.close()
    
    session.clear()
    return redirect('/')

# ================================
# API ENDPOINTS - LOCAL CITIZEN
# ================================

@app.route('/api/local/register', methods=['POST'])
def api_local_register():
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM locals WHERE email = ?', (data['email'],))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Email already exists'})
        
        cursor.execute('''
            INSERT INTO locals (name, email, phone, password, area)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['name'], data['email'], data['phone'], hash_password(data['password']), data['area']))
        conn.commit()
        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/api/local/login', methods=['POST'])
def api_local_login():
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, email, area, password FROM locals WHERE email = ?', 
                       (data['email'],))
        user = cursor.fetchone()
        if user and verify_password(user[4], data['password']):
            if not user[4].startswith(('pbkdf2:', 'scrypt:')):
                cursor.execute('UPDATE locals SET password = ? WHERE id = ?',
                               (hash_password(data['password']), user[0]))
            session['user_id'] = user[0]
            session['user_type'] = 'local'
            session['user_name'] = user[1]
            
            cursor.execute('INSERT INTO user_sessions (user_type, user_id, user_name) VALUES (?, ?, ?)', 
                          ('local', user[0], user[1]))
            conn.commit()
            return jsonify({'success': True, 'user_id': user[0], 'name': user[1], 'area': user[3]})
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/api/complaint/submit', methods=['POST'])
def submit_complaint():
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT name FROM locals WHERE id = ?', (data['local_id'],))
        local = cursor.fetchone()
        local_name = local[0] if local else 'Unknown'
        
        cursor.execute('''
            INSERT INTO complaints (local_id, local_name, location, sub_location, symptoms, water_issues, description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['local_id'], local_name, data['location'], data['sub_location'], 
              json.dumps(data['symptoms']), json.dumps(data['water_issues']), data['description'], 'pending'))
        conn.commit()
        return jsonify({'success': True, 'message': 'Complaint submitted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/api/complaints/<int:local_id>', methods=['GET'])
def get_user_complaints(local_id):
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, location, sub_location, symptoms, water_issues, description, status, created_at
            FROM complaints WHERE local_id = ? ORDER BY created_at DESC
        ''', (local_id,))
        complaints = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': c[0],
            'location': c[1],
            'sub_location': c[2],
            'symptoms': json.loads(c[3]) if c[3] else [],
            'water_issues': json.loads(c[4]) if c[4] else [],
            'description': c[5],
            'status': c[6],
            'created_at': c[7]
        } for c in complaints])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================================
# API ENDPOINTS - ASHA WORKER
# ================================

@app.route('/api/asha/register', methods=['POST'])
def api_asha_register():
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM asha_workers WHERE worker_id = ? OR email = ?', 
                      (data['worker_id'], data['email']))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Worker ID or Email already exists'})
        
        cursor.execute('''
            INSERT INTO asha_workers (name, worker_id, email, phone, password, area)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['name'], data['worker_id'], data['email'], data['phone'], hash_password(data['password']), data['area']))
        conn.commit()
        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/api/asha/login', methods=['POST'])
def api_asha_login():
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, worker_id, area, password FROM asha_workers WHERE worker_id = ?', 
                       (data['worker_id'],))
        user = cursor.fetchone()
        if user and verify_password(user[4], data['password']):
            if not user[4].startswith(('pbkdf2:', 'scrypt:')):
                cursor.execute('UPDATE asha_workers SET password = ? WHERE id = ?',
                               (hash_password(data['password']), user[0]))
            session['asha_id'] = user[0]
            session['user_type'] = 'asha'
            session['user_name'] = user[1]
            
            cursor.execute('INSERT INTO user_sessions (user_type, user_id, user_name) VALUES (?, ?, ?)', 
                          ('asha', user[0], user[1]))
            conn.commit()
            return jsonify({'success': True, 'asha_id': user[0], 'name': user[1], 'area': user[3]})
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'ML Model not loaded'}), 500
        
        data = request.json
        
        # Validate input ranges
        if not (20 <= data['temperature'] <= 45):
            return jsonify({'error': 'Temperature must be between 20-45°C'}), 400
        if not (6.5 <= data['ph'] <= 8.5):
            return jsonify({'error': 'pH must be between 6.5-8.5'}), 400
        if not (0 <= data['turbidity'] <= 500):
            return jsonify({'error': 'Turbidity must be between 0-500 NTU'}), 400
        if not (0 <= data['bacterial_count'] <= 1000):
            return jsonify({'error': 'Bacterial Count must be between 0-1000 CFU/mL'}), 400
        
        # Prepare input for prediction
        input_data = np.array([[
            data['temperature'], data['rainfall'], data['ph'], data['turbidity'],
            data['dissolved_oxygen'], data['nitrate'], data['lead'], data['bacterial_count'],
            data['clean_water_percentage'], data['sanitation_level'], data['healthcare_access'],
            data['symptom_diarrhea'], data['symptom_fever'], data['water_dirty'], data['water_scarcity']
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Convert prediction to string
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        result = risk_levels.get(prediction, 'Medium')
        
        print(f"Prediction value: {prediction}, Result: {result}")  # Debug log
        
        # Get ASHA name
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM asha_workers WHERE id = ?', (data['asha_id'],))
        asha = cursor.fetchone()
        asha_name = asha[0] if asha else 'Unknown'
        
        # Save prediction to database
        cursor.execute('''
            INSERT INTO predictions (
                asha_id, asha_name, temperature, rainfall, ph, turbidity, dissolved_oxygen,
                nitrate, lead, bacterial_count, clean_water_percentage, sanitation_level,
                healthcare_access, symptom_diarrhea, symptom_fever, water_dirty,
                water_scarcity, prediction, location, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ''', (data['asha_id'], asha_name, data['temperature'], data['rainfall'], data['ph'], 
              data['turbidity'], data['dissolved_oxygen'], data['nitrate'], data['lead'],
              data['bacterial_count'], data['clean_water_percentage'], data['sanitation_level'],
              data['healthcare_access'], data['symptom_diarrhea'], data['symptom_fever'],
              data['water_dirty'], data['water_scarcity'], result, data['location']))
        conn.commit()
        conn.close()
        
        # ========== EMAIL ALERT SYSTEM ==========
        # Send email alerts based on risk level
        try:
            if result == 'High':
                # Send high priority alert with detailed report
                send_risk_alert(asha_name, data['location'], result, data)
                print(f"✅ HIGH RISK email alert sent for {data['location']}")
                
                # Optional: Send to multiple recipients
                additional_recipients = [
                    'district_health_officer@health.gov.in',
                    'emergency_response@disaster.in'
                ]
                # for recipient in additional_recipients:
                #     send_email(recipient, subject, body)
                
            elif result == 'Medium':
                # Send medium risk notification
                send_medium_risk_alert(asha_name, data['location'], result, data)
                print(f"⚠️ MEDIUM RISK notification sent for {data['location']}")
                
            else:  # Low risk
                # Optional: Send low risk update (commented by default)
                # send_low_risk_update(asha_name, data['location'], data)
                print(f"✅ LOW RISK - No alert triggered for {data['location']}")
                
        except Exception as email_error:
            print(f"⚠️ Email notification failed but prediction succeeded: {email_error}")
            # Don't fail the prediction if email fails
        # ========================================
        
        return jsonify({'prediction': result})
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/api/asha/predictions', methods=['GET'])
def get_asha_predictions():
    try:
        asha_id = request.args.get('asha_id')
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, temperature, rainfall, ph, turbidity, prediction, location, created_at
            FROM predictions WHERE asha_id = ? ORDER BY created_at DESC
        ''', (asha_id,))
        predictions = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': p[0], 
            'temperature': p[1], 
            'rainfall': p[2], 
            'ph': p[3],
            'turbidity': p[4], 
            'prediction': p[5], 
            'location': p[6], 
            'date': p[7]
        } for p in predictions])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/asha/complaints', methods=['GET'])
def get_asha_complaints():
    try:
        area = request.args.get('area', '').strip()
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()

        if area:
            cursor.execute('''
                SELECT id, local_name, location, sub_location, symptoms, water_issues, description, status, created_at
                FROM complaints
                WHERE location = ?
                ORDER BY created_at DESC
                LIMIT 100
            ''', (area,))
        else:
            cursor.execute('''
                SELECT id, local_name, location, sub_location, symptoms, water_issues, description, status, created_at
                FROM complaints
                ORDER BY created_at DESC
                LIMIT 100
            ''')

        complaints = cursor.fetchall()
        conn.close()

        return jsonify([{
            'id': c[0],
            'local_name': c[1],
            'location': c[2],
            'sub_location': c[3],
            'symptoms': json.loads(c[4]) if c[4] else [],
            'water_issues': json.loads(c[5]) if c[5] else [],
            'description': c[6],
            'status': c[7],
            'created_at': c[8]
        } for c in complaints])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================================
# API ENDPOINTS - ADMIN (FIXED)
# ================================

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    try:
        data = request.json
        if data['username'] == 'admin' and data['password'] == 'admin123':
            session['admin'] = True
            session['user_type'] = 'admin'
            session['user_name'] = 'Administrator'
            
            conn = sqlite3.connect('database/water_disease.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO user_sessions (user_type, user_id, user_name) VALUES (?, ?, ?)', 
                          ('admin', 0, 'Administrator'))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/predictions', methods=['GET'])
def get_all_predictions():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions ORDER BY created_at DESC
        ''')
        predictions = cursor.fetchall()
        conn.close()
        
        result = []
        for p in predictions:
            # Correct column indices based on table schema
            # Index 18 is prediction, Index 19 is location, Index 20 is created_at
            prediction_value = p[18] if len(p) > 18 else 'Medium'
            location_value = p[19] if len(p) > 19 else 'Unknown'
            
            # Ensure prediction is a string
            if isinstance(prediction_value, (int, float)):
                risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                prediction_value = risk_map.get(prediction_value, 'Medium')
            
            result.append({
                'id': p[0],
                'asha_id': p[1],
                'asha_name': p[2],
                'temperature': p[3],
                'rainfall': p[4],
                'ph': p[5],
                'turbidity': p[6],
                'dissolved_oxygen': p[7],
                'nitrate': p[8],
                'lead': p[9],
                'bacterial_count': p[10],
                'clean_water_percentage': p[11],
                'sanitation_level': p[12],
                'healthcare_access': p[13],
                'symptom_diarrhea': p[14],
                'symptom_fever': p[15],
                'water_dirty': p[16],
                'water_scarcity': p[17],
                'prediction': prediction_value,
                'location': location_value,
                'date': p[20] if len(p) > 20 else 'Unknown'
            })
        
        print(f"Returning {len(result)} predictions")  # Debug
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_all_predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ... (rest of your admin endpoints remain the same)

@app.route('/api/admin/complaints', methods=['GET'])
def get_all_complaints():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM complaints ORDER BY created_at DESC
        ''')
        complaints = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': c[0],
            'local_id': c[1],
            'local_name': c[2],
            'location': c[3],
            'sub_location': c[4],
            'symptoms': json.loads(c[5]) if c[5] else [],
            'water_issues': json.loads(c[6]) if c[6] else [],
            'description': c[7],
            'status': c[8],
            'date': c[9]
        } for c in complaints])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/locals', methods=['GET'])
def get_all_locals():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, email, phone, area, created_at FROM locals ORDER BY created_at DESC')
        locals_data = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': l[0],
            'name': l[1],
            'email': l[2],
            'phone': l[3],
            'area': l[4],
            'created_at': l[5]
        } for l in locals_data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/asha', methods=['GET'])
def get_all_asha():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, worker_id, email, phone, area, created_at FROM asha_workers ORDER BY created_at DESC')
        asha_data = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': a[0],
            'name': a[1],
            'worker_id': a[2],
            'email': a[3],
            'phone': a[4],
            'area': a[5],
            'created_at': a[6]
        } for a in asha_data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/sessions', methods=['GET'])
def get_sessions():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM user_sessions ORDER BY login_time DESC
        ''')
        sessions = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'id': s[0],
            'user_type': s[1],
            'user_id': s[2],
            'user_name': s[3],
            'login_time': s[4],
            'logout_time': s[5]
        } for s in sessions])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/update_complaint_status/<int:id>', methods=['PUT'])
def update_complaint_status(id):
    try:
        data = request.json
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE complaints SET status = ? WHERE id = ?', (data['status'], id))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/admin/delete/<string:type>/<int:id>', methods=['DELETE'])
def delete_item(type, id):
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        if type == 'prediction':
            cursor.execute('DELETE FROM predictions WHERE id = ?', (id,))
        elif type == 'complaint':
            cursor.execute('DELETE FROM complaints WHERE id = ?', (id,))
        elif type == 'local':
            cursor.execute('DELETE FROM complaints WHERE local_id = ?', (id,))
            cursor.execute('DELETE FROM locals WHERE id = ?', (id,))
        elif type == 'asha':
            cursor.execute('DELETE FROM predictions WHERE asha_id = ?', (id,))
            cursor.execute('DELETE FROM asha_workers WHERE id = ?', (id,))
        else:
            return jsonify({'success': False, 'message': 'Invalid type'}), 400
        
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    try:
        conn = sqlite3.connect('database/water_disease.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = "High"')
        high_risk = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = "Medium"')
        medium_risk = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE prediction = "Low"')
        low_risk = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM complaints')
        total_complaints = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM complaints WHERE status = "pending"')
        pending_complaints = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM locals')
        total_locals = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM asha_workers')
        total_asha = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'total_complaints': total_complaints,
            'pending_complaints': pending_complaints,
            'total_locals': total_locals,
            'total_asha': total_asha
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ================================
# MAIN
# ================================

if __name__ == '__main__':
    os.makedirs('database', exist_ok=True)
    os.makedirs('ml_model', exist_ok=True)
    
    print("=" * 50)
    print("Water-Borne Disease Surveillance System")
    print("=" * 50)
    print(f"Server running at: http://localhost:5000")
    print(f"Admin credentials: admin / admin123")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)