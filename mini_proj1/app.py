from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import tensorflow as tf
import joblib
from functools import wraps
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Load the trained RNN model and scaler
model = tf.keras.models.load_model('rnn_model.h5')
scaler = joblib.load('scaler.pkl')

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT NOT NULL,
         email TEXT UNIQUE NOT NULL,
         password TEXT NOT NULL)
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('prediction_form'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ? AND password = ?', (email, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('prediction_form'))
        return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                     (name, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('signup.html', error='Email already exists')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/prediction-form')
@login_required
def prediction_form():
    return render_template('cancer_mutation.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            int_features = []
            for feature in request.form.values():
                try:
                    int_features.append(float(feature))
                except ValueError:
                    return render_template('cancer_mutation.html', 
                                        pred="Error: Please enter valid numeric values for all fields.")
            
            # Check if the number of inputs matches the expected count
            if len(int_features) != 14:
                return render_template('cancer_mutation.html', 
                                    pred="Error: Please provide exactly 14 inputs.")

            # Reshape and scale the input
            final_input = np.array(int_features).reshape(1, -1)
            final_input = scaler.transform(final_input).reshape(1, 1, len(int_features))

            # Make prediction
            prediction = model.predict(final_input)
            output = '{0:.{1}f}'.format(prediction[0][0], 2)

            if float(output) > 0.5:
                return render_template('cancer_mutation.html', 
                                    pred=f"Probability of analysed mutation is: {output} and there is risk of cancer.",
                                    bhai="Act quickly!")
            else:
                return render_template('cancer_mutation.html', 
                                    pred=f"Probability of analysed mutation is: {output}. The risk of cancer is low.",
                                    bhai="The mutation is not cancerous.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return render_template('cancer_mutation.html', 
                                pred="Error: An unexpected error occurred. Please check your input values.")
    
    return render_template('cancer_mutation.html')

if __name__ == "__main__":
    app.run(debug=True)