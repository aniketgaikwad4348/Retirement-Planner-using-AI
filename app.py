from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from flask_login import login_required
# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100), nullable=False)
    middlename = db.Column(db.String(100), nullable=True)
    lastname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    profile_photo_url = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f"User('{self.firstname}', '{self.lastname}', '{self.email}')"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create all tables
with app.app_context():
    db.create_all()

# Load and preprocess data (example data preprocessing function)
def preprocess_data(financial_data, user_data, investment_data):
    financial_data.ffill(inplace=True)
    user_data.ffill(inplace=True)
    investment_data.ffill(inplace=True)

    # Encode categorical variables
    user_data_encoded = pd.get_dummies(user_data, columns=['risk_tolerance', 'preferred_investment'])
    return financial_data, user_data_encoded, investment_data

# Example data loading
financial_data = pd.read_csv('historical_financial_data.csv')
user_data = pd.read_csv('user_data.csv')
investment_data = pd.read_csv('historical_investment_data.csv')

financial_data, user_data_encoded, investment_data = preprocess_data(financial_data, user_data, investment_data)

# Prepare data for modeling (example code, adjust as needed)
X = user_data_encoded.drop(columns=['required_savings'])
y = user_data_encoded['required_savings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model for savings prediction (example code, adjust as needed)
savings_model = LinearRegression()
savings_model.fit(X_train, y_train)

# Prepare investment training data (example code, adjust as needed)
investment_data_encoded = pd.get_dummies(investment_data, columns=['preferred_investment'])
investment_features = ['age', 'income', 'current_savings', 'risk_tolerance_Low', 'risk_tolerance_Medium', 'risk_tolerance_High']

for feature in investment_features:
    if feature not in investment_data_encoded.columns:
        investment_data_encoded[feature] = 0

investment_X = investment_data_encoded[investment_features]
investment_y = investment_data_encoded['market_return']

# Train model for investment recommendations (example code, adjust as needed)
investment_model = RandomForestRegressor(n_estimators=100, random_state=42)
investment_model.fit(investment_X, investment_y)

# Function to suggest top investments based on user's preferred choice (example function, adjust as needed)
def suggest_top_investments(preferred_investment):
    if preferred_investment == 'stocks':
        return [
            {'name': 'AAPL', 'trend_score': 0.9, 'expected_growth': 0.15},
            {'name': 'GOOGL', 'trend_score': 0.85, 'expected_growth': 0.12},
            {'name': 'MSFT', 'trend_score': 0.88, 'expected_growth': 0.14}
        ]
    elif preferred_investment == 'bonds':
        return [
            {'name': 'US Treasury Bonds', 'trend_score': 0.75, 'expected_growth': 0.06},
            {'name': 'Corporate Bonds', 'trend_score': 0.72, 'expected_growth': 0.05},
            {'name': 'Municipal Bonds', 'trend_score': 0.78, 'expected_growth': 0.07}
        ]
    elif preferred_investment == 'real_estate':
        return [
            {'name': 'Commercial Real Estate', 'trend_score': 0.8, 'expected_growth': 0.10},
            {'name': 'Residential Real Estate', 'trend_score': 0.76, 'expected_growth': 0.09},
            {'name': 'Industrial Real Estate', 'trend_score': 0.78, 'expected_growth': 0.11}
        ]
    elif preferred_investment == 'mutual_funds':
        return [
            {'name': 'Vanguard  SM&I Fund', 'trend_score': 0.82, 'expected_growth': 0.13},
            {'name': 'Fidelity Contrafund', 'trend_score': 0.79, 'expected_growth': 0.11},
            {'name': 'SPDR S&P 500 ETF Trust', 'trend_score': 0.85, 'expected_growth': 0.14}
        ]
    else:
        return []

# Routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user_input = request.form.to_dict()
    user_input = {key: float(value) if key not in ['risk_tolerance', 'preferred_investment'] else value for key, value in user_input.items()}

    user_input_encoded = pd.DataFrame([user_input])
    user_input_encoded = pd.get_dummies(user_input_encoded)

    missing_cols = set(X_train.columns) - set(user_input_encoded.columns)
    for col in missing_cols:
        user_input_encoded[col] = 0
    user_input_encoded = user_input_encoded[X_train.columns]

    user_input_array = user_input_encoded.values
    prediction = savings_model.predict(user_input_array)[0]

    preferred_investment = user_input.get('preferred_investment')
    suggested_investments = suggest_top_investments(preferred_investment)

    return render_template('result.html', required_savings=prediction, suggested_investments=suggested_investments)
@app.route('/about')
def about():
    return render_template('about.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        firstname = request.form['firstname']
        middlename = request.form['middlename']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']

        # Check if the email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('This email is already registered. Please use a different email or log in.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(firstname=firstname, middlename=middlename, lastname=lastname, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Your account has been created! You are now able to log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Update user route
@app.route('/update_user', methods=['POST'])
@login_required
def update_user():
    user = current_user
    if request.method == 'POST':
        user.firstname = request.form['firstname']
        user.middlename = request.form['middlename']
        user.lastname = request.form['lastname']
        db.session.commit()
        flash('Your profile has been updated successfully!', 'success')
    return redirect(url_for('profile'))

# Upload photo route
@app.route('/upload_photo', methods=['POST'])
@login_required
def upload_photo():
    user = current_user
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            photo.save(os.path.join(app.root_path, 'static', 'profile_photos', photo.filename))
            user.profile_photo_url = url_for('static', filename='profile_photos/' + photo.filename)
            db.session.commit()
            flash('Your profile photo has been updated!', 'success')
    return redirect(url_for('profile'))

# Profile route
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

if __name__ == '__main__':
    app.run(debug=True)
