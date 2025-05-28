# SFBT Terapist Chatbotu (Solution-Focused Brief Therapy AI Assistant)

Bu proje, çözüm odaklı kısa terapi (SFBT) modeline dayalı, kullanıcılarla terapi benzeri etkileşimler kuran bir yapay zeka asistanıdır. PDF formatındaki terapi dokümanlarını analiz eder, embedding yöntemiyle dizin oluşturur ve OpenAI LLM (GPT) ile doğal bir diyalog gerçekleştirir.

---

##  Özellikler

- **PDF tabanlı bilgi çıkarımı:** PyMuPDF (fitz) ile içerik ayrıştırma
- **Vektör tabanlı bilgi arama:** SentenceTransformer + FAISS
- **Retrieval-Augmented Generation (RAG):** Belge destekli GPT yanıtları
- **Tematik soru bankası:** “İlişki”, “İş”, “Özgüven” temalarında SFBT’ye özgü sorular
- **Kullanıcı profili & seans özetleri:** `dataclass` yapısıyla veri yönetimi

---

##  Mimarî Yapı
PDF Belgeleri
↓
PyMuPDF ile bölme
↓
SentenceTransformer ile embedding
↓
FAISS ile vektör indeksleme
↓ ↓
Kullanıcı girişi → Tema & soru seçimi
↓
OpenAI GPT → Cevap üretimi
##  Teknolojiler

| Kütüphane              | Açıklama                                      |
|------------------------|-----------------------------------------------|
| `fitz` (PyMuPDF)       | PDF dosyalarını bölütleme                     |
| `sentence-transformers`| Cümle embedding (MiniLM)                      |
| `faiss`                | Vektör diziniyle arama                        |
| `openai`               | GPT-3.5 veya GPT-4 API entegrasyonu           |
| `dotenv`               | API anahtarı yönetimi                         |
| `dataclass`            | Kullanıcı ve seans verileri                   |

---

## Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
OPENAI_API_KEY=your_openai_api_key
sfbt_terapi.pdf

iliski_terapi.pdf

is_stres_terapi.pdf

terapi.pdf

🧩 Kod Bileşenleri
UserProfile
python
Kopyala
Düzenle
@dataclass
class UserProfile:
    name: str
    age: Optional[int]
    primary_concerns: List[str]
    session_count: int
    total_scaling_average: float
SessionSummary
Seans içeriği, ölçekleme skorları, hedefler gibi verileri tutar.

ThematicQuestionBank
Tematik ve kategorik SFBT sorularını barındırır:

miracle_questions

scaling_questions

exception_questions

SFBTBot
PDF içeriklerini işler, FAISS dizini oluşturur, kullanıcı temelli soru yöneltir ve GPT ile yanıt üretir.

Örnek Kullanım
python
Kopyala
Düzenle
bot = SFBTBot()
bot.user_profile = UserProfile(name="Ali", age=28, primary_concerns=["özgüven"])
question = bot.question_bank.get_question("özgüven", "scaling_questions")
print("Bot sorusu:", question)

Geliştirici Notları
FAISS dizini dışa aktarılabilir.
Embedding cache yapılabilir.
Test senaryolarında sahte kullanıcılar kullanılabilir.
GPT çağrılarında tema bağlamı korunur.

Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
