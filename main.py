from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from datetime import datetime
from collections import defaultdict
import re
import matplotlib
matplotlib.use('Agg')  # Backend for server
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Optional
from dataclasses import dataclass

# Request modeli
class ChatRequest(BaseModel):
    message: str

# OpenAI API Key
API_KEY = "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"
os.environ['OPENAI_API_KEY'] = API_KEY

@dataclass
class UserProfile:
    name: str = ""
    age: Optional[int] = None
    primary_concerns: List[str] = None
    session_count: int = 0
    total_scaling_average: float = 0.0
    
    def __post_init__(self):
        if self.primary_concerns is None:
            self.primary_concerns = []

@dataclass
class SessionSummary:
    session_id: int
    date: str
    main_topics: List[str]
    goals_identified: List[str]
    exceptions_found: List[str]
    scaling_scores: Dict[str, int]
    interventions_used: List[str]
    progress_notes: str

class ThematicQuestionBank:
    def __init__(self):
        self.themes = {
            "ilişki": {
                "miracle_questions": [
                    "Yarın sabah uyandığınızda ilişkinizde mucizevi bir iyileşme olmuş olsaydı, bunu ilk nasıl fark ederdiniz?",
                    "Eşiniz/partneriniz sizde mucizevi bir değişim gözlemlese, bunu nasıl anlayacağını düşünüyorsunuz?",
                    "İlişkinizde en çok istediğiniz değişim bir gecede olsaydı, ertesi gün evinizde atmosfer nasıl olurdu?"
                ],
                "scaling_questions": [
                    "İlişkinizde mutluluk seviyenizi 1-10 arasında nasıl değerlendirirsiniz?",
                    "Partnerinizle iletişim kalitenizi 1-10 skalasında nereye koyarsınız?",
                    "İlişkinizde güven seviyenizi 1-10 arasında puanlasanız?"
                ],
                "exception_questions": [
                    "Son zamanlarda partnerinizle aranızın daha iyi olduğu anlar hangileri?",
                    "Birbirinizi daha çok anladığınız zamanlar nasıl geçiyor?",
                    "İlişkinizde size umut veren küçük anlar hangileri?"
                ]
            },
            "iş": {
                "miracle_questions": [
                    "Yarın işe gittiğinizde mucizevi bir değişim olmuş olsaydı, bunu ilk nasıl hissederdiniz?",
                    "İş hayatınızda en çok istediğiniz şey bir gecede gerçekleşseydi, iş yerindeki davranışlarınız nasıl değişirdi?"
                ],
                "scaling_questions": [
                    "İş tatmininizi 1-10 arasında nasıl değerlendirirsiniz?",
                    "İş-yaşam dengenizi 1-10 arasında puanlasanız?"
                ],
                "exception_questions": [
                    "İşte kendinizi daha başarılı hissettiğiniz günler nasıl geçiyor?",
                    "İş stresinin daha az olduğu zamanlar ne yapıyorsunuz farklı?"
                ]
            },
            "özgüven": {
                "miracle_questions": [
                    "Yarın uyandığınızda kendinize olan güveniniz mükemmel seviyede olsaydı, günün ilk saatlerinde nasıl davranırdınız?",
                    "Özgüveninizle ilgili mucizevi bir iyileşme olsaydı, çevrenizdekiler bunu nasıl anlardı?"
                ],
                "scaling_questions": [
                    "Özgüven seviyenizi 1-10 arasında nereye koyarsınız?",
                    "Kendinizi değerli hissetme derecenizi 1-10 ile puanlasanız?"
                ],
                "exception_questions": [
                    "Kendinizi güçlü ve değerli hissettiğiniz anlar hangileri?",
                    "Özgüveninizin daha yüksek olduğu durumlar nasıl oluyor?"
                ]
            }
        }
    
    def get_question(self, theme: str, question_type: str) -> str:
        if theme not in self.themes:
            theme = "özgüven"
        if question_type not in self.themes[theme]:
            question_type = "scaling_questions"
        
        questions = self.themes[theme][question_type]
        return np.random.choice(questions)

class SFBTBot:
    def __init__(self):
        print("SFBT Bot başlatılıyor...")
        
        # OpenAI client
        self.openai_client = openai.OpenAI()
        
        # Embedding modeli
        try:
            print("Embedding modeli yükleniyor...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Embedding modeli yüklendi")
        except Exception as e:
            print(f"Embedding modeli yüklenemedi: {e}")
            self.embedding_model = None
        
        # User profili ve session verisi
        self.user_profile = UserProfile()
        self.conversation_history = []
        self.current_session_data = {
            'themes': defaultdict(int),
            'scaling_scores': {},
            'goals': [],
            'exceptions': [],
            'interventions': []
        }
        
        # RAG sistemi
        self.pdf_chunks = []
        self.pdf_index = None
        
        # Soru bankası
        self.question_bank = ThematicQuestionBank()
        self.current_theme = None
        
        print("Bot temel yapısı hazır!")

    def load_pdfs(self):
        """PDF dosyalarını yükle"""
        pdf_folder = "pdf_files"
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
            print(f"'{pdf_folder}' klasörü oluşturuldu")
        
        pdf_files = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                pdf_files.append(os.path.join(pdf_folder, filename))
        
        if not pdf_files:
            print("PDF dosyası bulunamadı")
            return
        
        print(f"{len(pdf_files)} PDF yükleniyor...")
        
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                print(f"{os.path.basename(pdf_path)} işleniyor...")
                
                doc = fitz.open(pdf_path)
                text = ""
                
                # İlk 8 sayfayı oku (daha fazla içerik)
                for page_num in range(min(len(doc), 8)):
                    text += doc[page_num].get_text() + " "
                
                doc.close()
                
                if text.strip():
                    chunks = self._create_chunks(text, pdf_path)
                    all_chunks.extend(chunks)
                    print(f"  {len(chunks)} parça")
                
            except Exception as e:
                print(f"  Hata: {e}")
        
        # En fazla 40 chunk (daha fazla bilgi)
        if len(all_chunks) > 40:
            all_chunks = all_chunks[:40]
        
        self.pdf_chunks = all_chunks
        
        if self.pdf_chunks:
            self._create_index()

    def _create_chunks(self, text, source):
        """Text'i parçalara böl"""
        sentences = text.split('.')
        chunks = []
        current = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
                
            if len(current + sentence) > 300 and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        
        if current.strip():
            chunks.append(current.strip())
        
        return [f"[{os.path.basename(source)}] {chunk}" for chunk in chunks if len(chunk) > 50]

    def _create_index(self):
        """FAISS index oluştur"""
        try:
            embeddings = self.embedding_model.encode(self.pdf_chunks)
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            self.pdf_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.pdf_index.add(embeddings)
            
            print(f"{len(self.pdf_chunks)} chunk için index oluşturuldu")
            
        except Exception as e:
            print(f"Index hatası: {e}")

    def search_knowledge(self, query):
        """Bilgi ara"""
        if not self.pdf_index or not self.pdf_chunks or not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.pdf_index.search(query_embedding, 2)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if scores[0][i] > 0.25:  # Similarity threshold
                    results.append(self.pdf_chunks[idx])
            
            return results
            
        except Exception as e:
            print(f"Arama hatası: {e}")
            return []

    def detect_theme(self, user_input):
        """Tema tespit et"""
        text = user_input.lower()
        
        if any(word in text for word in ['eş', 'partner', 'sevgili', 'ilişki', 'evlilik', 'aşk']):
            return 'ilişki'
        elif any(word in text for word in ['iş', 'meslek', 'çalışma', 'patron', 'kariyer', 'şirket', 'mobbing']):
            return 'iş'
        elif any(word in text for word in ['özgüven', 'değersiz', 'kendimi', 'güvensiz', 'utanç', 'mutsuz']):
            return 'özgüven'
        else:
            return 'özgüven'

    def generate_response(self, user_input):
        """Ana yanıt üretimi"""
        theme = self.detect_theme(user_input)
        self.current_theme = theme
        self.current_session_data['themes'][theme] += 1
        
        knowledge = self.search_knowledge(user_input)
        
        system_msg = f"""Sen deneyimli bir SFBT (Çözüm Odaklı Kısa Süreli Terapi) uzmanısın.

YAKLAŞIMIN:
- Önce empati göster, sonra çözüm odaklı yönlendir
- Danışanın duygularını doğrula
- Güçlü yanlarını keşfet ve vurgula
- Somut sorular sor VE pratik öneriler ver
- Umut verici ama gerçekçi ol
- Türkçe konuş ve samimi ol

TEMA: {theme.upper()}

YAPIT TARZI:
1. Empati ile başla ("Anlıyorum ki...")
2. Güçlü yanını fark et ("Bu durumda bile...")  
3. Çözüm odaklı soru sor
4. Somut bir öneri/teknik ver
5. Umutla bitir

ÖNEMLİ: Detaylı, kapsamlı ve yönlendirici yanıtlar ver! Emoji kullanma."""
        
        messages = [{"role": "system", "content": system_msg}]
        
        if knowledge:
            knowledge_text = "\n".join(knowledge[:2])
            messages.append({"role": "system", "content": f"Bu kaynak bilgiyi yanıtında kullan:\n{knowledge_text[:400]}"})
        
        # Context için son 2 mesaj
        if len(self.conversation_history) >= 2:
            for msg in self.conversation_history[-2:]:
                messages.append({"role": "user", "content": msg['user']})
                messages.append({"role": "assistant", "content": msg['bot']})
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature=0.8,
                presence_penalty=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI hatası: {e}")
            return self._get_fallback(theme)

    def _get_fallback(self, theme):
        """Yedek yanıtlar"""
        fallbacks = {
            'ilişki': """
Anlıyorum ki ilişkinizde zorlu bir dönem geçiriyorsunuz. İlişki problemleri gerçekten yıpratıcı olabiliyor, ama buraya gelip bu konuyu konuşmaya cesaret etmeniz çözüm arayışınızı gösteriyor.

İlişkinizde neyin farklı olmasını istersiniz? Partnerinizle aranızda daha iyi geçen anlar hangileri? 

Küçük bir teknik önerebilirim: Bugün partnerinize bir pozitif mesaj gönderin veya onun yaptığı küçük bir şeye teşekkür edin. Küçük jestler büyük değişimlerin başlangıcı olabilir.

Bu değişim için hangi küçük adımı atmaya hazırsınız?""",
            
            'iş': """
İş hayatındaki zorluklarınızı anlıyorum. Mobbing ve iş stresi gerçekten çok yıpratıcı olabiliyor. Ama bu konuda farkındalık sahibi olmanız ve çözüm aramanız güçlü yanınızı gösteriyor.

İş hayatınızda en çok hangi alanda iyileşme görmek istersiniz? Kendinizi daha güçlü hissettiğiniz iş anları var mı?

İş memnuniyetinizi 1-10 skalasında değerlendirirseniz kendinizi nerede görüyorsunuz?

Pratik öneri: Her gün bir küçük başarınızı not edin. Bu, motivasyon ve pozitif odaklanma için çok etkili bir tekniktir.

Hangi alanda ilk değişikliği yapmak istersiniz?""",
            
            'özgüven': """
Mutsuzluk ve özgüven konusundaki mücadelenizi anlıyorum. Kendimizi değersiz hissetmek çok acı verici, ama bu duyguları fark etmeniz ve değiştirmek istemeniz aslında büyük bir güç göstergesi.

Kendinizi güçlü ve değerli hissettiğiniz anları hatırlıyor musunuz? O zamanlarda neler farklıydı?

Önerim: Her gün üç şeyi yazın - bir başarınız, bir güçlü yanınız, bir teşekkür ettiğiniz şey. Bu alıştırma beynin pozitif odaklanmasını güçlendirir.

Bu konuda hangi küçük adımı atmaya hazırsınız?"""
        }
        
        return fallbacks.get(theme, """
Bu konudaki zorluğunuzu anlıyorum. Her zorluk aynı zamanda değişim ve büyüme fırsatıdır.

Bu durumda neler değişse hayatınız daha iyi olurdu? Küçük değişiklikler büyük farklar yaratabilir.

Ben sizinle bu yolculukta yanınızdayım.""")

    def _handle_score(self, score):
        """Skor değerlendirmesi"""
        theme = self.current_theme or 'genel'
        self.current_session_data['scaling_scores'][theme] = score
        
        if score <= 3:
            return f"""{score}/10 - Bu gerçekten zor bir dönem geçiriyorsunuz.

Bu kadar zorlu durumda bile {score} puan vermeniz, içinizde güçlü bir dayanıklılık olduğunu gösteriyor.

Sizi bu {score} puanda tutan şey nedir? 
{score+1} puana çıkmak için hangi küçük adım en kolay olurdu?"""

        elif score <= 6:
            return f"""{score}/10 - Orta seviyede bir durum.

Bu {score} puanı veren olumlu şeyler neler?
Daha önce {score+1} veya {score+2} puana çıktığınız zamanlar oldu mu?
Bir sonraki seviyeye çıkmak için ne gerekli olabilir?"""

        else:
            return f"""{score}/10 - Harika bir seviye! Tebrikler!

Bu başarıyı nasıl elde ettiniz?
Bu yüksek puanı koruyan şey nedir?
Bu başarılı yaklaşımınızı hayatınızın başka alanlarına nasıl taşıyabilirsiniz?"""

    def show_progress_chart(self):
        """İlerleme grafiğini oluştur"""
        try:
            if not self.current_session_data['scaling_scores']:
                return {"error": "Henüz scaling verisi yok. Önce bir tema için 1-10 arası puanlama yapın."}
            
            # Grafik oluştur
            plt.figure(figsize=(10, 6))
            plt.style.use('default')
            
            themes = list(self.current_session_data['scaling_scores'].keys())
            scores = list(self.current_session_data['scaling_scores'].values())
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = plt.bar(themes, scores, color=colors[:len(themes)], alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            plt.title('TEMA BAZINDA İLERLEME DURUMU', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Skor (1-10)', fontsize=12, fontweight='bold')
            plt.xlabel('Temalar', fontsize=12, fontweight='bold')
            plt.ylim(0, 10)
            
            # Değerleri bar üzerinde göster
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}/10', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.gca().set_facecolor('#f8f9fa')
            plt.tight_layout()
            
            # Base64'e çevir
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            graphic = base64.b64encode(image_png).decode()
            
            # Analiz metni
            avg_score = np.mean(scores)
            max_theme = themes[scores.index(max(scores))]
            min_theme = themes[scores.index(min(scores))]
            
            analysis = f"""
GRAFİK ANALİZİ:

Ortalama Skor: {avg_score:.1f}/10
En Yüksek: {max_theme.title()} ({max(scores)}/10)
Gelişim Alanı: {min_theme.title()} ({min(scores)}/10)

ÖNERİ: {min_theme.title()} teması üzerinde çalışmaya odaklanabilirsiniz.
Hedef: Tüm alanlarda 7+ puana ulaşmak"""
            
            return {"graphic": graphic, "analysis": analysis}
            
        except Exception as e:
            return {"error": f"Grafik oluşturma hatası: {e}"}

    def show_session_chart(self):
        """Seans istatistikleri grafiği"""
        try:
            if not self.current_session_data['themes']:
                return {"error": "Henüz tema verisi yok."}
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            themes = list(self.current_session_data['themes'].keys())
            counts = list(self.current_session_data['themes'].values())
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            
            # Sol grafik: Tema dağılımı
            ax1.pie(counts, labels=themes, autopct='%1.1f%%', startangle=90,
                   colors=colors[:len(themes)], explode=[0.05]*len(themes))
            ax1.set_title('Tema Dağılımı', fontweight='bold')
            
            # Sağ grafik: Mesaj sayıları
            ax2.bar(themes, counts, color=colors[:len(themes)], alpha=0.7)
            ax2.set_title('Tema Başına Mesaj Sayısı', fontweight='bold')
            ax2.set_ylabel('Mesaj Sayısı')
            
            # Değerleri göster
            for i, count in enumerate(counts):
                ax2.text(i, count + 0.1, str(count), ha='center', fontweight='bold')
            
            plt.tight_layout()
            
            # Base64'e çevir
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            graphic = base64.b64encode(image_png).decode()
            
            return {"graphic": graphic, "summary": f"Toplam {sum(counts)} mesaj, {len(themes)} farklı tema"}
            
        except Exception as e:
            return {"error": f"Grafik hatası: {e}"}

    def get_summary(self):
        """Seans özeti"""
        if not self.current_session_data['themes']:
            return "Henüz konuşma başlamadı. Hangi konuda yardım istersiniz?"
        
        summary = "SEANS ÖZETİ\n\n"
        
        # Temalar
        summary += "Konuşulan Temalar:\n"
        for theme, count in self.current_session_data['themes'].items():
            summary += f"   • {theme.title()}: {count} mesaj\n"
        
        # Skorlar
        if self.current_session_data['scaling_scores']:
            summary += "\nVerilen Skorlar:\n"
            for theme, score in self.current_session_data['scaling_scores'].items():
                summary += f"   • {theme.title()}: {score}/10\n"
        
        summary += f"\nToplam: {len(self.conversation_history)} mesaj"
        
        return summary

    def get_response(self, message):
        """FastAPI için ana method"""
        if not message.strip():
            return "Size nasıl yardımcı olabilirim?"
        
        user_lower = message.lower().strip()
        
        # Özel komutlar
        if user_lower == 'temalar':
            return "TEMALAR:\n• İlişki\n• İş\n• Özgüven\n\nHangi konuda konuşmak istersiniz?"
        
        elif user_lower == 'özet':
            return self.get_summary()
        
        elif user_lower == 'mucize':
            q = self.question_bank.get_question(self.current_theme or 'özgüven', 'miracle_questions')
            return f"MUCIZE SORUSU:\n\n{q}"
        
        elif user_lower == 'derece':
            q = self.question_bank.get_question(self.current_theme or 'özgüven', 'scaling_questions')
            return f"DERECELENDIRME:\n\n{q}"
        
        elif user_lower == 'grafik':
            return "grafik_data"  # Özel işaret
        
        elif user_lower == 'istatistik':
            return "istatistik_data"  # Özel işaret
        
        elif user_lower in ['yardım', 'help', 'komutlar']:
            return """
SFBT BOT KOMUTLARI:

TEMEL KOMUTLAR:
'temalar' - Tema seçenekleri
'mucize' - Mucize sorusu
'derece' - Derecelendirme sorusu
'özet' - Seans özeti

GRAFİK KOMUTLARI:
'grafik' - İlerleme grafiği (scaling skorları)
'istatistik' - Seans istatistikleri

SİSTEM KOMUTLARI:
'yardım' - Bu yardım menüsü

İPUCU: Herhangi bir konuda 1-10 arası puan verirseniz otomatik değerlendirme yaparım!
"""
        
        # Basit selamlaşma
        if any(g in user_lower for g in ["merhaba", "selam"]):
            return "Merhaba! Ben SFBT Terapi Botuyum. Size nasıl yardımcı olabilirim?"
        
        if "teşekkür" in user_lower:
            return "Rica ederim! Başka bir konuda yardımcı olabilirim mi?"
        
        # Skor kontrolü
        score_match = re.search(r'\b([1-9]|10)\b', message)
        if score_match and ('puan' in user_lower or '/10' in message or 'derece' in user_lower):
            score = int(score_match.group(1))
            return self._handle_score(score)
        
        # Normal yanıt
        response = self.generate_response(message)
        
        # Kaydet
        self.conversation_history.append({
            'user': message,
            'bot': response,
            'timestamp': datetime.now().strftime("%H:%M"),
            'theme': self.current_theme
        })
        
        return response

# Bot örneği oluştur
bot = SFBTBot()

# Lifespan yönetimi
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Uygulama başlatılıyor...")
    bot.load_pdfs()
    print("Bot hazır!")
    yield
    print("Uygulama kapatılıyor...")

app = FastAPI(lifespan=lifespan)

# CORS Ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "SFBT Bot API çalışıyor!", "status": "OK", "version": "3.0"}

# API Endpoint'leri
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = bot.get_response(request.message)
        
        # Özel grafik yanıtları
        if response == "grafik_data":
            chart_data = bot.show_progress_chart()
            return {"response": "İlerleme grafiği oluşturuldu", "chart_data": chart_data, "status": "success"}
        
        elif response == "istatistik_data":
            chart_data = bot.show_session_chart()
            return {"response": "Seans istatistikleri oluşturuldu", "chart_data": chart_data, "status": "success"}
        
        return {"response": response, "status": "success"}
    except Exception as e:
        return {"response": "Bir hata oluştu, lütfen tekrar deneyin.", "error": str(e), "status": "error"}

# Grafik endpoint'i
@app.get("/api/progress")
async def get_progress():
    try:
        chart_data = bot.show_progress_chart()
        return chart_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/statistics")
async def get_statistics():
    try:
        chart_data = bot.show_session_chart()
        return chart_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/summary")
async def get_session_summary():
    try:
        summary = bot.get_summary()
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

# Server başlatma
if __name__ == "__main__":
    import uvicorn
    print("Tam Özellikli SFBT Server başlatılıyor...")
    print("API: http://127.0.0.1:8002")
    print("Docs: http://127.0.0.1:8002/docs")
    print("Durdurmak için Ctrl+C basın")
    
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8002, log_level="info")
    except KeyboardInterrupt:
        print("\nServer kapatıldı!")
    except Exception as e:
        print(f"Server hatası: {e}")