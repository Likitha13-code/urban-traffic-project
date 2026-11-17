# Urban Traffic Management: Pattern Mining and Congestion Prediction

## Run
    pip install -r requirements.txt
    python pipeline.py                 # synthetic 120-day hourly data, 8 zones
    python pipeline.py --csv my.csv    # real data: timestamp, area, hour, dow, weekend, rain_mm, event, volume,
                                       #   Car, Two-wheeler, Bus, Truck, Auto-rickshaw, avg_speed, congestion_idx
Open `dashboard.html` in a browser (needs internet for Chart.js CDN).

## Pipeline (pipeline.py)
1. Preprocessing: dedupe, interpolate missing values, IQR outlier clipping, congestion levels, lag features (24h, 168h), encoding, scaling.
2. Pattern mining: Apriori (support, confidence, lift) for rules leading to High/Severe congestion.
3. Classification: Random Forest, Decision Tree, k-NN (accuracy, weighted F1, confusion matrix).
4. Regression: Gradient Boosting, Random Forest, Linear (R2, MAE, RMSE).
5. Clustering: K-Means on zone-hour profiles, k picked by silhouette score.
6. Evaluation uses a chronological split: last 21 days are held out.

## Weka / Orange / Shogun
- Weka: open `data/traffic.arff`, choose class `level`, run J48 / RandomForest / Apriori (discretize first) and compare to results.json.
- Orange: File widget -> `data/traffic_clean.csv` -> Test and Score / k-Means / Association Rules add-on.
- Shogun: k-NN runs automatically if `shogun` is importable and is added to the classifier table.

## Note on data
The bundled dataset is synthetic (generated from realistic peak-hour profiles). Accuracy figures describe this data; re-run with a real dataset before reporting results.
