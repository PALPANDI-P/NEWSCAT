from backend.models.simple_classifier import SimpleNewsClassifier
c = SimpleNewsClassifier()

tests = [
    ('NASA launches new Mars rover mission to search for signs of ancient life on the red planet', 'space'),
    ('Apple unveils the iPhone 16 with a faster A18 chip and improved camera system', 'consumer_electronics'),
    ('The Federal Reserve raised interest rates by 25 basis points amid stubborn inflation', 'economy'),
    ('LeBron James scored 40 points as the Lakers defeated the Celtics in overtime', 'basketball'),
    ('Scientists use CRISPR gene editing to cure sickle cell disease in clinical trial', 'genetics'),
    ('Breaking: 7.2 magnitude earthquake hits coast of Japan, tsunami warning issued', 'breaking_news'),
    ('New study finds Mediterranean diet reduces heart disease risk by 30 percent', 'nutrition'),
    ('Russia-Ukraine conflict escalates as new military offensive begins in eastern front', 'war_conflict'),
    ('Netflix announces record subscriber growth driven by ad-supported tier', 'streaming'),
    ('Bitcoin surges past 100000 dollars reaching all time high amid institutional adoption', 'cryptocurrency'),
]

correct = 0
for text, expected in tests:
    result = c.classify(text, include_confidence=True)
    cat = result['category']
    conf = result['confidence']
    match = '✓' if cat == expected else '✗'
    if cat == expected:
        correct += 1
    print(f'{match} Expected: {expected:25s} Got: {cat:25s} Conf: {conf:.1f}%  Text: {text[:60]}...')

print(f'\nAccuracy: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)')
