
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import json
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or 'yazarlar-platformu-gizli-anahtar'
app.config['DATABASE'] = 'yazarlar_platformu.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'


def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                user_type TEXT NOT NULL CHECK(user_type IN ('yazar', 'izleyici', 'misafir')),
                full_name TEXT,
                bio TEXT,
                profile_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        ''')
        
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                author_id INTEGER NOT NULL,
                category_id INTEGER,
                tags TEXT,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES users (id),
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        ''')
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS likes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(article_id, user_id),
                FOREIGN KEY (article_id) REFERENCES articles (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        
        default_categories = [
            ('Roman', 'Kurgu romanları ve hikayeler'),
            ('Şiir', 'Şiir ve nazım eserleri'),
            ('Deneme', 'Düşünce ve deneme yazıları'),
            ('Bilim Kurgu', 'Bilim kurgu eserleri'),
            ('Fantastik', 'Fantastik edebiyat'),
            ('Tarih', 'Tarihi eserler ve araştırmalar'),
            ('Biyografi', 'Yaşam öyküleri'),
            ('Kişisel Gelişim', 'Kişisel gelişim yazıları'),
            ('Teknoloji', 'Teknoloji ve dijital dünya'),
            ('Sanat', 'Sanat ve estetik üzerine yazılar')
        ]
        
        for category in default_categories:
            cursor.execute('INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)', category)
        
        db.commit()


class AIAssistant:
    def __init__(self):
        self.stop_words = set(stopwords.words('turkish'))
        
    def generate_summary(self, text, max_sentences=3):
        """Metinden özet oluştur"""
        try:
            
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            else:
                return ' '.join(sentences[:max_sentences]) + '...'
        except:
            return text[:200] + '...' if len(text) > 200 else text
    
    def find_similar_articles(self, article_content, all_articles, top_n=5):
        """Benzer makaleleri bul"""
        try:
           
            contents = [article_content] + [art['content'] for art in all_articles]
            
            
            vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
            tfidf_matrix = vectorizer.fit_transform(contents)
            
           
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            
            similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
            
           
            similar_articles = []
            for idx in similar_indices:
                if cosine_similarities[idx] > 0.1: 
                    similar_articles.append({
                        'article': all_articles[idx],
                        'similarity_score': float(cosine_similarities[idx])
                    })
            
            return similar_articles
        except Exception as e:
            print(f"AI hatası: {e}")
            return []
    
    def suggest_category(self, title, content):
        """İçeriğe göre kategori öner"""
       
        keywords = {
            'roman': ['roman', 'hikaye', 'kurgu', 'kahraman'],
            'şiir': ['şiir', 'dize', 'kafiye', 'nazım'],
            'bilim kurgu': ['uzay', 'gelecek', 'teknoloji', 'robot', 'alien'],
            'tarih': ['tarih', 'geçmiş', 'savaş', 'osmanlı', 'cumhuriyet'],
            'kişisel gelişim': ['gelişim', 'başarı', 'motivasyon', 'hedef']
        }
        
        text = (title + ' ' + content).lower()
        scores = {}
        
        for category, words in keywords.items():
            score = sum(1 for word in words if word in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'Diğer'

ai_assistant = AIAssistant()


@app.route('/')
def index():
    if 'user_id' in session:
        db = get_db()
        
       
        popular_articles = db.execute('''
            SELECT a.*, u.username, u.full_name, c.name as category_name 
            FROM articles a 
            JOIN users u ON a.author_id = u.id 
            LEFT JOIN categories c ON a.category_id = c.id 
            ORDER BY a.views DESC LIMIT 5
        ''').fetchall()
        
        
        new_articles = db.execute('''
            SELECT a.*, u.username, u.full_name, c.name as category_name 
            FROM articles a 
            JOIN users u ON a.author_id = u.id 
            LEFT JOIN categories c ON a.category_id = c.id 
            ORDER BY a.created_at DESC LIMIT 5
        ''').fetchall()
        
        
        categories = db.execute('SELECT * FROM categories LIMIT 10').fetchall()
        
        return render_template('index.html', 
                             popular_articles=popular_articles,
                             new_articles=new_articles,
                             categories=categories,
                             user_type=session.get('user_type'))
    
    return render_template('welcome.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        full_name = request.form.get('full_name', '')
        
        if not username or not email or not password:
            flash('Lütfen tüm alanları doldurunuz.', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        
        db = get_db()
        try:
            cursor = db.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password, user_type, full_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, hashed_password, user_type, full_name))
            db.commit()
            
            flash('Kayıt başarılı! Lütfen giriş yapın.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            flash('Bu kullanıcı adı veya email zaten kullanılıyor.', 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_type'] = user['user_type']
            session['full_name'] = user['full_name']
            
            flash('Başarıyla giriş yaptınız!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Kullanıcı adı veya şifre hatalı.', 'error')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('index'))

@app.route('/article/create', methods=['GET', 'POST'])
def create_article():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))
    
    if session.get('user_type') != 'yazar':
        flash('Sadece yazarlar makale oluşturabilir.', 'error')
        return redirect(url_for('index'))
    
    db = get_db()
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        category_id = request.form.get('category_id')
        tags = request.form.get('tags', '')
        
       
        summary = ai_assistant.generate_summary(content)
        
       
        suggested_category = None
        if not category_id or category_id == '':
            suggested_category = ai_assistant.suggest_category(title, content)
            
            if suggested_category:
                cat = db.execute('SELECT id FROM categories WHERE name LIKE ?', 
                               (f'%{suggested_category}%',)).fetchone()
                if cat:
                    category_id = cat['id']
        
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO articles (title, content, summary, author_id, category_id, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, content, summary, session['user_id'], category_id, tags))
        
        article_id = cursor.lastrowid
        db.commit()
        
        flash('Makaleniz başarıyla oluşturuldu!', 'success')
        return redirect(url_for('view_article', article_id=article_id))
    
    
    categories = db.execute('SELECT * FROM categories ORDER BY name').fetchall()
    return render_template('create_article.html', categories=categories)


@app.route('/article/<int:article_id>')
def view_article(article_id):
    db = get_db()
    
   
    db.execute('UPDATE articles SET views = views + 1 WHERE id = ?', (article_id,))
    
   
    article = db.execute('''
        SELECT a.*, u.username, u.full_name, u.profile_image, 
               c.name as category_name, c.id as category_id
        FROM articles a 
        JOIN users u ON a.author_id = u.id 
        LEFT JOIN categories c ON a.category_id = c.id 
        WHERE a.id = ?
    ''', (article_id,)).fetchone()
    
    if not article:
        flash('Makale bulunamadı.', 'error')
        return redirect(url_for('index'))
    
    
    comments = db.execute('''
        SELECT c.*, u.username, u.full_name, u.profile_image
        FROM comments c 
        JOIN users u ON c.user_id = u.id 
        WHERE c.article_id = ?
        ORDER BY c.created_at DESC
    ''', (article_id,)).fetchall()
    
   
    similar_articles = []
    if article['content']:
      
        same_category_articles = db.execute('''
            SELECT a.*, u.username, u.full_name 
            FROM articles a 
            JOIN users u ON a.author_id = u.id 
            WHERE a.category_id = ? AND a.id != ?
            LIMIT 10
        ''', (article['category_id'], article_id)).fetchall()
        
        
        articles_list = [dict(art) for art in same_category_articles]
        
       
        similar_articles = ai_assistant.find_similar_articles(
            article['content'], 
            articles_list,
            top_n=3
        )
    
   
    user_liked = False
    if 'user_id' in session:
        like = db.execute('SELECT id FROM likes WHERE article_id = ? AND user_id = ?', 
                         (article_id, session['user_id'])).fetchone()
        user_liked = like is not None
    
    db.commit()
    return render_template('view_article.html', 
                         article=article, 
                         comments=comments,
                         similar_articles=similar_articles,
                         user_liked=user_liked)


@app.route('/articles')
def articles():
    category_id = request.args.get('category_id')
    search = request.args.get('search', '')
    
    db = get_db()
    query = '''
        SELECT a.*, u.username, u.full_name, c.name as category_name 
        FROM articles a 
        JOIN users u ON a.author_id = u.id 
        LEFT JOIN categories c ON a.category_id = c.id 
        WHERE 1=1
    '''
    params = []
    
    if category_id and category_id != 'all':
        query += ' AND a.category_id = ?'
        params.append(category_id)
    
    if search:
        query += ' AND (a.title LIKE ? OR a.content LIKE ? OR a.tags LIKE ?)'
        search_term = f'%{search}%'
        params.extend([search_term, search_term, search_term])
    
    query += ' ORDER BY a.created_at DESC'
    
    articles_list = db.execute(query, params).fetchall()
    categories = db.execute('SELECT * FROM categories ORDER BY name').fetchall()
    
    return render_template('articles.html', 
                         articles=articles_list,
                         categories=categories,
                         selected_category=category_id,
                         search=search)


@app.route('/article/<int:article_id>/comment', methods=['POST'])
def add_comment(article_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Lütfen önce giriş yapın.'}), 401
    
    content = request.form.get('content')
    if not content or len(content.strip()) < 3:
        return jsonify({'error': 'Yorum en az 3 karakter olmalıdır.'}), 400
    
    db = get_db()
    db.execute('''
        INSERT INTO comments (article_id, user_id, content)
        VALUES (?, ?, ?)
    ''', (article_id, session['user_id'], content.strip()))
    db.commit()
    
    return jsonify({'success': True})


@app.route('/article/<int:article_id>/like', methods=['POST'])
def toggle_like(article_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Lütfen önce giriş yapın.'}), 401
    
    db = get_db()
    
    existing_like = db.execute('SELECT id FROM likes WHERE article_id = ? AND user_id = ?', 
                              (article_id, session['user_id'])).fetchone()
    
    if existing_like:
        
        db.execute('DELETE FROM likes WHERE id = ?', (existing_like['id'],))
        db.execute('UPDATE articles SET likes = likes - 1 WHERE id = ?', (article_id,))
        liked = False
    else:
        
        db.execute('INSERT INTO likes (article_id, user_id) VALUES (?, ?)', 
                  (article_id, session['user_id']))
        db.execute('UPDATE articles SET likes = likes + 1 WHERE id = ?', (article_id,))
        liked = True
    
    db.commit()
    
   
    article = db.execute('SELECT likes FROM articles WHERE id = ?', (article_id,)).fetchone()
    
    return jsonify({'liked': liked, 'likes_count': article['likes']})


@app.route('/profile/<username>')
def profile(username):
    db = get_db()
    
  
    user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if not user:
        flash('Kullanıcı bulunamadı.', 'error')
        return redirect(url_for('index'))
    
   
    articles = db.execute('''
        SELECT a.*, c.name as category_name 
        FROM articles a 
        LEFT JOIN categories c ON a.category_id = c.id 
        WHERE a.author_id = ? 
        ORDER BY a.created_at DESC
    ''', (user['id'],)).fetchall()
    
    return render_template('profile.html', user=user, articles=articles)


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Metin gerekli'}), 400
    
    summary = ai_assistant.generate_summary(text)
    return jsonify({'summary': summary})


@app.route('/api/suggest_category', methods=['POST'])
def api_suggest_category():
    if 'user_id' not in session:
        return jsonify({'error': 'Yetkisiz erişim'}), 401
    
    data = request.get_json()
    title = data.get('title', '')
    content = data.get('content', '')
    
    if not title and not content:
        return jsonify({'error': 'Başlık veya içerik gerekli'}), 400
    
    suggested = ai_assistant.suggest_category(title, content)
    
    db = get_db()
    categories = db.execute('SELECT * FROM categories WHERE name LIKE ?', 
                           (f'%{suggested}%',)).fetchall()
    
    if categories:
        return jsonify({
            'suggested_category': suggested,
            'categories': [dict(cat) for cat in categories]
        })
    else:
        return jsonify({'suggested_category': suggested})


@app.route('/category/<int:category_id>/top')
def top_articles_by_category(category_id):
    db = get_db()
    
   
    category = db.execute('SELECT * FROM categories WHERE id = ?', (category_id,)).fetchone()
    if not category:
        flash('Kategori bulunamadı.', 'error')
        return redirect(url_for('articles'))
    
   
    articles = db.execute('''
        SELECT a.*, u.username, u.full_name 
        FROM articles a 
        JOIN users u ON a.author_id = u.id 
        WHERE a.category_id = ?
        ORDER BY (a.likes * 0.3 + a.views * 0.7) DESC
        LIMIT 10
    ''', (category_id,)).fetchall()
    
    
    best_article = None
    if articles:
       
        articles_list = list(articles)
        if len(articles_list) > 0:
            best_article = articles_list[0]
    
    return render_template('top_articles.html',
                         category=category,
                         articles=articles,
                         best_article=best_article)


import os

if not os.path.exists('templates'):
    os.makedirs('templates')


welcome_html = '''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yazarlar Platformu - Hoş Geldiniz</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: #4f46e5;
            text-decoration: none;
        }
        
        .logo span {
            color: #7c3aed;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        
        .nav-links a {
            color: #4b5563;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-links a:hover {
            color: #4f46e5;
        }
        
        .btn {
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary {
            background: #4f46e5;
            color: white;
        }
        
        .btn-primary:hover {
            background: #4338ca;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: white;
            color: #4f46e5;
            border: 2px solid #4f46e5;
        }
        
        .btn-secondary:hover {
            background: #f3f4f6;
        }
        
        .welcome-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 4rem 2rem;
            text-align: center;
            color: white;
        }
        
        .hero {
            max-width: 800px;
            margin-bottom: 3rem;
        }
        
        .hero h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            font-weight: 800;
        }
        
        .hero p {
            font-size: 1.3rem;
            margin-bottom: 2rem;
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .features {
            display: flex;
            gap: 2rem;
            margin-top: 3rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 16px;
            width: 300px;
            transition: transform 0.3s;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #a78bfa;
        }
        
        .feature-card h3 {
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .footer {
            background: rgba(0, 0, 0, 0.2);
            color: white;
            text-align: center;
            padding: 1.5rem;
            margin-top: auto;
        }
        
        .cta-buttons {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
        }
        
        @media (max-width: 768px) {
            .navbar {
                flex-direction: column;
                gap: 1rem;
            }
            
            .nav-links {
                gap: 1rem;
            }
            
            .hero h1 {
                font-size: 2.5rem;
            }
            
            .cta-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="/" class="logo">Yaz<span>Platform</span></a>
        <div class="nav-links">
            <a href="#features">Özellikler</a>
            <a href="#about">Hakkında</a>
        </div>
        <div>
            <a href="/login" class="btn btn-secondary">Giriş</a>
            <a href="/register" class="btn btn-primary">Kayıt Ol</a>
        </div>
    </nav>
    
    <div class="welcome-container">
        <div class="hero">
            <h1>Yazarlar İçin Modern Platform</h1>
            <p>Eserlerinizi paylaşın, yapay zeka destekli öneriler alın ve diğer yazarlarla bağlantı kurun. LinkedIn benzeri deneyimle yazarlık becerilerinizi geliştirin.</p>
            <div class="cta-buttons">
                <a href="/register" class="btn btn-primary" style="font-size: 1.1rem; padding: 0.8rem 2rem;">
                    <i class="fas fa-user-plus"></i> Hemen Katıl
                </a>
                <a href="/articles" class="btn btn-secondary" style="font-size: 1.1rem; padding: 0.8rem 2rem;">
                    <i class="fas fa-book-open"></i> Makaleleri Keşfet
                </a>
            </div>
        </div>
        
        <div class="features" id="features">
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <h3>Yapay Zeka Destekli</h3>
                <p>Makaleleriniz için otomatik özet oluşturma ve kategori önerileri</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-users"></i>
                </div>
                <h3>Yazar Topluluğu</h3>
                <p>Benzer ilgi alanlarına sahip yazarlarla bağlantı kurun</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <h3>Analitik ve İstatistikler</h3>
                <p>Eserlerinizin performansını takip edin ve geliştirin</p>
            </div>
        </div>
    </div>
    
    <div class="footer" id="about">
        <p>© 2023 Yazarlar Platformu - Tüm hakları saklıdır</p>
        <p>Yapay zeka destekli yazarlık topluluğu</p>
    </div>
</body>
</html>'''

index_html = '''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yazarlar Platformu - Ana Sayfa</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f9fafb;
            color: #333;
        }
        
        .navbar {
            background: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: #4f46e5;
            text-decoration: none;
        }
        
        .logo span {
            color: #7c3aed;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-links a {
            color: #4b5563;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-links a:hover {
            color: #4f46e5;
        }
        
        .user-menu {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #4f46e5;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .btn {
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary {
            background: #4f46e5;
            color: white;
        }
        
        .btn-primary:hover {
            background: #4338ca;
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .btn-danger:hover {
            background: #dc2626;
        }
        
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
        }
        
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem;
            border-radius: 16px;
            margin-bottom: 2rem;
        }
        
        .hero h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .hero p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        
        .ai-feature {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 4px solid #4f46e5;
        }
        
        .ai-feature h3 {
            color: #4f46e5;
            margin-bottom: 0.5rem;
        }
        
        .ai-feature p {
            color: #6b7280;
            margin-bottom: 1rem;
        }
        
        .sections {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }
