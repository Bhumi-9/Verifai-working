"""
VerifAI — Train Model Script (No-NLTK version)
Trains a Logistic Regression on an embedded dataset.
Run once: python3 train_model.py
Outputs: model/model.pkl and model/vectorizer.pkl
"""

import pickle, re, os, string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── INLINE STOPWORDS (no NLTK needed) ─────────────────────
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'is','it','its','was','are','be','been','has','have','had','do','does',
    'did','will','would','could','should','may','might','shall','can','need',
    'i','me','my','we','our','you','your','he','his','she','her','they','them',
    'this','that','these','those','what','which','who','how','when','where',
    'not','no','nor','so','yet','both','either','neither','just','very','too',
    'also','more','most','any','all','some','such','even','than','then','there',
    'here','about','above','after','before','between','into','through','during',
    'if','as','up','out','off','over','under','again','further'
}

def clean(text: str) -> str:
    """Clean text without NLTK."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)         # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# ── EMBEDDED TRAINING DATA ────────────────────────────────
# Label: 1 = Fake, 0 = Genuine
# ~200 carefully crafted examples covering typical patterns

FAKE_REVIEWS = [
    "BEST PRODUCT EVER!!! Absolutely amazing!!! Everyone NEEDS this now!!!",
    "OMG I love love love this so much!!! Changed my life!!! Buy it immediately!!!",
    "WOW 5 stars!!! Perfect in every single way!!! Must buy for everyone!!!",
    "Amazing product!! Best purchase of my ENTIRE life!! Tell all your friends!!",
    "This is INCREDIBLE!! I can't believe how good this is!! 10/10 no doubt!!!",
    "Perfect!!! PERFECT!!! I have never been so happy with a purchase!! BUY NOW!!!",
    "Absolutely fantastic product!! Best I've ever seen in my life!! Love it!!!",
    "LOVE LOVE LOVE!! Best thing ever made!! Everyone should own one of these!!!",
    "Outstanding quality!! Super amazing!! This changed everything for me!! WOW!!",
    "Greatest product on the market!! Nothing compares!! Buy buy buy!!!",
    "Incredible!! I showed all my friends and they ordered one too!! AMAZING!!!",
    "Best seller for a reason!! Perfect quality!! Don't hesitate just buy now!!!",
    "I am obsessed!! This is hands down the best product in the world!! Wow!!!",
    "Simply PERFECT!! No flaws whatsoever!! Every person alive needs this product!",
    "OMG OMG OMG I love this so much!! Best brand ever!! 100 stars if I could!!",
    "FANTASTIC quality!! I cried tears of joy when I received this!! BUY IT!!!",
    "Nothing short of a miracle product!! Absolutely life-changing!! 5 stars!!!",
    "ABSOLUTELY PERFECT!! I have told every single person I know to buy this!!",
    "This product is PURE GOLD!! Amazing amazing amazing!! Must have item!!!",
    "Super duper amazing!! The best thing I have ever bought online!! Wow wow wow!!",
    "Zero complaints absolutely none at all!! Perfect in every way possible!!",
    "Best ever!! Love it to pieces!! Can't live without it!! Everyone needs this!!",
    "Ordered 10 of these for family!! So good!! Best product on Amazon hands down!",
    "100% recommend!! Best in class!! Nothing else comes close!! Pure perfection!!",
    "I have purchased many products but this one is by far the most perfect one!!",
    "Exceeded every expectation!! Absolutely flawless product!! Love love love!!!",
    "I rate this product 100 out of 10!! Simply the best thing ever invented!!!",
    "Forget everything else!! This is the only product you will ever need!! WOW!",
    "STUNNING quality!! I immediately bought three more for gifts!! Perfect item!!",
    "This is what perfection looks like!! Zero issues whatsoever!! BUY THIS NOW!",
    "Amazing fast shipping great product love it so much best ever 5 stars always",
    "Perfect product perfect seller perfect transaction will buy again 100 percent",
    "Great great great amazing amazing amazing product love love love it so much",
    "Highly highly highly recommend this product to everyone you know buy now!!!",
    "Do yourself a favor and buy this immediately you will not regret it at all!!",
    "Stunning beautiful perfect magnificent exceptional wonderful 5 stars forever!!",
    "I have never written a review but this product is SO GOOD I had to share!!",
    "Blew my mind!! Absolutely blew my mind!! This is the product of the century!!",
    "Every single feature works perfectly!! Not one single flaw!! Pure perfection!!",
    "The best product I have ever had the pleasure of buying!! Life changing!!!",
    "Just wow. Wow wow wow. I am speechless. This is the most perfect thing ever.",
    "Superb quality amazing price everyone needs to order this right now trust me!",
    "Received as gift and immediately ordered 5 more amazing quality top notch!!!",
    "Game changer! Life changer! Mind changer! Everything changer! Best ever made!",
    "So good I returned the other brand and bought 4 of these instead!! Perfect!!",
    "This product is FIRE!!! Absolutely on fire best thing since sliced bread!!!",
    "Rating: Infinite stars. Product: Perfect. Seller: Perfect. Everything: Perfect.",
    "I've been searching for this my whole life and now I found it!! Amazing!!!",
    "It cured all my problems!! Made everything better!! Perfect product forever!!",
    "INSANE quality!! I can't get over how amazing this is!! Tell everyone!!!",
]

GENUINE_REVIEWS = [
    "Decent quality for the price. The stitching came loose after about 3 washes, but customer service sent a replacement without hassle.",
    "Works as advertised. Battery life is roughly 6 hours with moderate use. Nothing spectacular but gets the job done.",
    "I've been using this for about 3 months now. It's held up reasonably well, though the color faded a bit faster than I'd expected.",
    "Good value for the money. Packaging was slightly damaged on arrival, but the product itself was completely fine.",
    "Fits true to size. The material feels decent, though not as premium as the photos suggest. Would buy again at this price point.",
    "Does exactly what it says. Took about 12 days to arrive. Setup took maybe 20 minutes and was pretty straightforward.",
    "Comfortable enough for everyday use. The zipper feels a bit flimsy, but the main compartment is spacious and well-organized.",
    "Pretty good overall. My main complaint is the instructions are poorly written — took some trial and error to figure out the settings.",
    "Solid build quality. I've dropped it twice and no damage so far. The screen is a bit dim outdoors but fine indoors.",
    "Mixed feelings. The product works well but the size was smaller than the listing implied. Would've ordered the next size up.",
    "Reasonably good. The first one arrived with a cracked corner piece and they replaced it within a week. Second one is fine.",
    "It works but I wish it came with better documentation. Had to watch a YouTube video to understand the advanced settings.",
    "Used it for 6 weeks before writing this. Still working as expected. Charging cable feels cheaply made though.",
    "Exactly what I needed for the price. Not luxury quality but not claiming to be either. Functional and durable so far.",
    "Took longer than expected to arrive (2.5 weeks) but the product quality is good once it showed up.",
    "Fits my 15-inch laptop easily with room for accessories. The shoulder strap padding could be thicker for heavier loads.",
    "Three months in and no major issues. The handle shows some wear but the main structure is intact. Reasonably satisfied.",
    "Good product but the color in person is slightly different from the website photos — more grey than blue in my case.",
    "Works great for light use. Started having intermittent connection issues around month 4, but a reset fixed it.",
    "Solid value. My only gripe is the packaging was excessive — lots of plastic waste for a small item.",
    "Better than I expected at this price point. Some rough edges on the casing but nothing that affects performance.",
    "Comfortable and lightweight. After about 2 weeks the strap started slipping but I adjusted it and it's been fine since.",
    "Does the job. Not the prettiest but functional. The buttons are a little stiff at first but loosened up after a week.",
    "Happy with the purchase overall. Customer support was helpful when I had a setup question, responded within a day.",
    "Average product. It does what you need it to do but don't expect anything beyond the basic functionality.",
    "Quite good for the cost. I've owned more expensive versions and this holds up fine for occasional home use.",
    "Works well for my purposes. The battery drains faster in cold weather but that's expected for this type of product.",
    "Reasonable quality. One of the clips broke after a month but the seller offered a partial refund which I appreciated.",
    "Better than the previous model I had. Setup was easy and performance has been reliable across about 50 uses so far.",
    "Decent everyday product. The cleaning was a bit involved the first time but gets easier. Instructions could be clearer.",
    "Good but not great. Would have liked a slightly longer warranty given the price. Works well for the first few months.",
    "Arrived on time, well packaged. The assembly required was more involved than expected — about 30 minutes to put together.",
    "I use this daily and it's held up fine over four months. Nothing exceptional but nothing to complain about either.",
    "Works as described. My daughter uses it for school and finds it comfortable. The colour options are limited though.",
    "Satisfactory product. Looks slightly different from the listing image but the quality is consistent with similar items.",
    "After 5 weeks of regular use, still working reliably. The noise level is acceptable and it's straightforward to clean.",
    "Good for the price I paid during the sale. Full price feels a bit steep for what it is, but at a discount it's worth it.",
    "Functional and well made. I ordered based on reviews here and they were accurate. No major surprises either way.",
    "Decent quality but the sizing runs small — order at least one size up from your usual. Comfortable once I exchanged it.",
    "Using it for 2 months now. The performance is consistent. Build quality is a 7 out of 10 — not premium but not flimsy.",
    "It does the job adequately. I've had better versions at a higher price and worse ones at the same price. About average.",
    "Solid product. Some assembly required but nothing difficult. The rubber grip is comfortable and hasn't worn down yet.",
    "Good for what I needed. Not designed for heavy-duty use but works perfectly for light to medium daily tasks.",
    "Three weeks in, no complaints. The texture feels different from the product images but the quality is fine.",
    "Reasonably impressed. The product came with a slight scratch on the surface but it doesn't affect how it works.",
    "Works exactly as expected. I've used it about 30 times now and see no signs of wear. Happy with the purchase.",
    "Adequate for the price. If you need something durable for heavy use, you might want to spend more. For casual use, fine.",
    "Takes a little getting used to but after a few days it's intuitive. Quality feels appropriate for the price range.",
    "I'm on my second one — the first lasted about 14 months of regular use. Good value overall, would buy again.",
    "Happy with it. The smell when first opened was a bit strong but dissipated after a day or two. Works well now.",
]

def build_dataset():
    texts, labels = [], []
    for r in FAKE_REVIEWS:
        texts.append(r)
        labels.append(1)
    for r in GENUINE_REVIEWS:
        texts.append(r)
        labels.append(0)
    # Augment with slight variations
    import random
    random.seed(42)
    extra_fake = [
        "The most perfect product I have ever purchased!!!",
        "Unbelievable quality!! Best thing I have ever owned!! 5 stars!!!",
        "SO GOOD!! Can't stop telling people about this!! AMAZING PRODUCT!!!",
        "Mind blowing purchase!! Perfect packaging perfect product perfect seller!!",
        "Don't waste time looking elsewhere this IS the best product available!!!",
    ] * 4
    extra_genuine = [
        "Good enough for casual use. Nothing fancy but reliable so far.",
        "Took longer to arrive than expected. Product itself is fine, no complaints.",
        "Works as described. Instructions were a bit confusing but figured it out.",
        "Decent quality for a budget option. Don't expect premium but it holds up.",
        "Has minor cosmetic defects out of box but functionally it's perfectly fine.",
    ] * 4
    for r in extra_fake:
        texts.append(r); labels.append(1)
    for r in extra_genuine:
        texts.append(r); labels.append(0)
    return texts, labels

if __name__ == '__main__':
    print("🚀 VerifAI — Training Model\n" + "─" * 45)

    texts, labels = build_dataset()
    cleaned = [clean(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        analyzer='word'
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc*100:.1f}%")
    print(classification_report(y_test, preds, target_names=['Genuine', 'Fake']))

    os.makedirs('model', exist_ok=True)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model/stopwords.pkl', 'wb') as f:
        pickle.dump(STOPWORDS, f)

    print("✅  Saved: model/model.pkl, model/vectorizer.pkl")
    print("   Run: python3 app.py")
