#!/usr/bin/env python
"""Test the classifier directly"""
import sys
sys.path.insert(0, 'e:/NEWSCAT')

try:
    from backend.models.simple_classifier import SimpleNewsClassifier
    clf = SimpleNewsClassifier()
    result = clf.classify('Apple reported record quarterly earnings today, beating analyst expectations.')
    print("Category:", result.get('category'))
    print("Confidence:", result.get('confidence'))
    print("Top predictions:", result.get('top_predictions'))
    print("Method:", result.get('method'))
    print("is_trained:", clf.is_trained)
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc()
