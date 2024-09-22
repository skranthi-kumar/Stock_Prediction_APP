from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key in production
login_manager = LoginManager()
login_manager.init_app(app)

# In-memory user storage for demonstration
users = {'admin': 'admin'}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            user = User(username)
            login_user(user)
            return redirect("http://127.0.0.1:5002/")  # Redirect to the prediction app's root URL
        else:
            flash("Invalid credentials")
            return render_template('login.html')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/protected')
@login_required
def protected():
    return "This is a protected route."

if __name__ == '__main__':
    app.run(port=5001)
