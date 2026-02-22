# News Classification Enhancement Plan

## Overview
This plan outlines the expansion of the NEWSCAT classification system from 20 categories to 35 categories, with improved keyword identification and better disambiguation between similar categories.

## Implementation Status: ✅ COMPLETED

---

## Changes Made

### 1. Configuration Updates ([`backend/config.py`](backend/config.py))
- Added 15 new categories to the `CATEGORIES` dictionary
- Maintained backward compatibility with all existing categories
- Total categories: 35

### 2. Classifier Updates ([`backend/models/optimized_classifier.py`](backend/models/optimized_classifier.py))
- Added comprehensive keyword dictionaries for 15 new categories
- Each category has high, medium, and low priority keywords
- Keywords include modern terminology and trending topics
- Regex patterns automatically compiled for faster matching

### 3. Training Data Updates ([`backend/data/training/news_samples.json`](backend/data/training/news_samples.json))
- Added 75 new training samples (5 per new category)
- Total training samples: 125
- Diverse representation of each category

---

## Categories Summary

### Existing Categories (20)
| Group | Categories |
|-------|------------|
| Core | technology, sports, politics, business, entertainment |
| Extended | health, science, world, education, environment |
| Specialized | finance, automotive, travel, food, fashion |
| Niche | realestate, legal, religion, lifestyle, opinion |

### New Categories Added (15)
| Group | Categories |
|-------|------------|
| Real Incident | accidents, crime, disasters, protests |
| Human-Centric | career, relationships, mentalhealth |
| Specialized News | investigative, breaking, weather |
| Additional | infrastructure, socialmedia, gaming, space, agriculture |

---

## Detailed Category Information

### Real Incident Categories

#### 1. `accidents`
**Description**: Traffic accidents, industrial incidents, crashes, derailments
**High Priority Keywords**: `plane crash`, `train derailment`, `car accident`, `traffic collision`, `industrial accident`, `workplace accident`, `fatal crash`, `multi-vehicle`, `pileup`
**Disambiguation**: Accidents are man-made/unintentional incidents; disasters are large-scale natural events

#### 2. `crime`
**Description**: Criminal activities, investigations, arrests, theft, assault
**High Priority Keywords**: `murder`, `homicide`, `robbery`, `burglary`, `assault`, `kidnapping`, `arson`, `fraud`, `embezzlement`, `organized crime`, `serial killer`, `mass shooting`
**Disambiguation**: Crime focuses on criminal acts; legal focuses on court proceedings

#### 3. `disasters`
**Description**: Natural disasters, earthquakes, floods, hurricanes, wildfires
**High Priority Keywords**: `earthquake`, `hurricane`, `tsunami`, `wildfire`, `tornado`, `flood`, `volcanic eruption`, `landslide`, `typhoon`, `cyclone`
**Disambiguation**: Disasters are large-scale events; accidents are smaller-scale incidents

#### 4. `protests`
**Description**: Demonstrations, rallies, civil unrest, activism, strikes
**High Priority Keywords**: `protest`, `demonstration`, `rally`, `march`, `strike`, `civil unrest`, `riot`, `uprising`, `activism`, `activist`, `boycott`
**Disambiguation**: Protests focus on demonstrations; politics focuses on government

---

### Human-Centric Categories

#### 5. `career`
**Description**: Jobs, employment, workplace, hiring, layoffs
**High Priority Keywords**: `job market`, `employment`, `unemployment rate`, `hiring`, `layoff`, `jobless claims`, `workforce`, `recruitment`
**Disambiguation**: Career focuses on individual employment; business focuses on companies

#### 6. `relationships`
**Description**: Dating, marriage, family dynamics, divorce, parenting
**High Priority Keywords**: `marriage`, `divorce`, `wedding`, `dating`, `relationship`, `engagement`, `anniversary`, `family`, `parenting`, `custody`
**Disambiguation**: Relationships focuses on interpersonal connections; lifestyle is broader

#### 7. `mentalhealth`
**Description**: Mental health awareness, psychology, therapy, depression
**High Priority Keywords**: `mental health`, `depression`, `anxiety`, `therapy`, `psychologist`, `psychiatrist`, `suicide prevention`, `ptsd`, `bipolar disorder`
**Disambiguation**: Mentalhealth specifically addresses psychological wellbeing; health is general medical

---

### Specialized News Types

#### 8. `investigative`
**Description**: In-depth reporting, exposés, whistleblower stories
**High Priority Keywords**: `investigation`, `expose`, `whistleblower`, `deep dive`, `special report`, `investigative journalism`, `uncovered`, `revealed`, `leaked documents`

#### 9. `breaking`
**Description**: Breaking news, urgent alerts, developing stories
**High Priority Keywords**: `breaking news`, `just in`, `developing story`, `live update`, `urgent`, `alert`, `emergency broadcast`

#### 10. `weather`
**Description**: Weather forecasts, storms, meteorological news
**High Priority Keywords**: `weather forecast`, `storm`, `hurricane`, `tornado warning`, `blizzard`, `heat wave`, `cold front`, `tropical storm`
**Disambiguation**: Weather is forecasting; disasters is actual catastrophic events

---

### Additional Categories

#### 11. `infrastructure`
**Description**: Construction, public works, utilities, transportation
**High Priority Keywords**: `infrastructure bill`, `bridge collapse`, `road construction`, `public works`, `utility outage`, `power grid`, `water main break`

#### 12. `socialmedia`
**Description**: Social media trends, platform news, viral content
**High Priority Keywords**: `viral`, `trending`, `tiktok`, `instagram`, `twitter`, `facebook`, `youtube`, `influencer`, `content creator`

#### 13. `gaming`
**Description**: Video games, esports, gaming industry
**High Priority Keywords**: `video game`, `esports`, `gaming`, `playstation`, `xbox`, `nintendo`, `pc gaming`, `game release`, `gaming tournament`

#### 14. `space`
**Description**: Space exploration, satellites, rockets, missions
**High Priority Keywords**: `spacex`, `nasa`, `rocket launch`, `space station`, `mars mission`, `moon landing`, `satellite`, `astronaut`
**Disambiguation**: Space is specifically about space exploration; science is broader

#### 15. `agriculture`
**Description**: Farming, crops, livestock, agricultural policy
**High Priority Keywords**: `farming`, `agriculture`, `crop`, `harvest`, `livestock`, `farm bill`, `agricultural`, `food production`

---

## Keyword Improvements

### Modern Terminology Added
- **Technology**: `chatgpt`, `openai`, `blockchain`, `cryptocurrency`, `nft`, `metaverse`
- **Social Media**: `tiktok`, `influencer`, `viral`, `content creator`, `hashtag`
- **Work**: `remote work`, `hybrid work`, `gig economy`, `freelance`
- **Mental Health**: `burnout`, `mindfulness`, `self-care`, `teletherapy`

### Disambiguation Context Words
| Category Pair | Distinguishing Keywords |
|---------------|------------------------|
| crime vs legal | `arrest`, `suspect`, `investigation` vs `court`, `verdict`, `attorney` |
| accidents vs disasters | `collision`, `crash`, `wreck` vs `earthquake`, `hurricane`, `evacuation` |
| protests vs politics | `demonstration`, `march`, `rally` vs `election`, `legislation`, `congress` |
| career vs business | `job`, `salary`, `hiring` vs `stock`, `revenue`, `quarterly` |

---

## Files Modified

| File | Changes |
|------|---------|
| [`backend/config.py`](backend/config.py) | Added 15 new categories to CATEGORIES dict |
| [`backend/models/optimized_classifier.py`](backend/models/optimized_classifier.py) | Added keyword dictionaries for 15 new categories |
| [`backend/data/training/news_samples.json`](backend/data/training/news_samples.json) | Added 75 new training samples |

---

## Backward Compatibility

✅ All existing categories preserved
✅ Existing keyword dictionaries unchanged
✅ New categories added without modifying existing structure
✅ Training data IDs continue sequentially (51-125)

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Total Categories | 20 | 35 |
| Training Samples | 50 | 125 |
| Keyword Entries | ~500 | ~1200 |

### New Categories Added: 15
1. accidents - Traffic and industrial incidents
2. crime - Criminal activities and investigations
3. disasters - Natural and large-scale catastrophes
4. protests - Demonstrations and civil unrest
5. career - Employment and workplace
6. relationships - Family and interpersonal
7. mentalhealth - Psychological wellbeing
8. investigative - In-depth reporting
9. breaking - Urgent news alerts
10. weather - Forecasts and conditions
11. infrastructure - Public works and utilities
12. socialmedia - Platform trends and viral content
13. gaming - Video games and esports
14. space - Space exploration
15. agriculture - Farming and food production
