from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your AdaBoost model
adaboost_model = joblib.load('adaboost_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting input data from the form and handling empty strings
        age_first_funding_year = float(request.form.get('ageFirstFunding', '0') or '0')
        age_last_funding_year = float(request.form.get('ageLastFunding', '0') or '0')
        age_first_milestone_year = float(request.form.get('ageFirstMilestone', '0') or '0')
        age_last_milestone_year = float(request.form.get('ageLastMilestone', '0') or '0')
        relationships = float(request.form.get('numRelationship', '0') or '0')
        funding_rounds = float(request.form.get('numFundingRounds', '0') or '0')
        funding_total_usd = float(request.form.get('totalFunding', '0') or '0')
        milestones = float(request.form.get('numMilestones', '0') or '0')
        avg_participants = float(request.form.get('numParticipants', '0') or '0')

        # Add placeholders for the remaining features
        additional_features = [0.0] * (39 - 9)  # Adjust based on your actual features

        # Create a list with the input values
        input_data = [
            age_first_funding_year,
            age_last_funding_year,
            age_first_milestone_year,
            age_last_milestone_year,
            relationships,
            funding_rounds,
            funding_total_usd,
            milestones,
            avg_participants
        ] + additional_features

        # Make a prediction using the loaded model
        prediction = adaboost_model.predict([input_data])[0]

        # If the model outputs probabilities or continuous values, use a threshold
        if not isinstance(prediction, int):
            prediction = 1 if prediction >= 0.5 else 0

        # Map the predicted label to a meaningful output
        if prediction == 1:
            result = 'Acquired'
        else:
            result = 'Closed'

        # Render the prediction result
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
