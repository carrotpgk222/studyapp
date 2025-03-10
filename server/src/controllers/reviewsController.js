// reviewsController.js
const reviewsModel = require('../models/reviewsModel');
const axios = require('axios');

function mapMinutesToStr(duration) {
  if (duration < 30) return 'Less than 30 minutes';
  if (duration < 60) return '30-60 minutes';
  if (duration < 120) return '1-2 hours';
  return 'More than 2 hours';
}

module.exports.createReview = (req, res) => {
  // Extract data from request body
  const data = {
    sessionDuration: req.body.sessionDuration,       // e.g. 45
    rating: req.body.rating,                         // e.g. "happy"
    feedback: req.body.feedback || '',               // optional feedback
    user_id: req.body.user_id,                       // which user wrote the review
    scheduleSatisfaction: req.body.scheduleSatisfaction // e.g. 3 (1–5)
  };

  // Validate required fields
  if (
    data.sessionDuration == null ||
    !data.rating ||
    !data.user_id ||
    data.scheduleSatisfaction == null
  ) {
    return res.status(400).json({ error: 'Missing required fields.' });
  }

  // 1) Insert the review into your DB
  reviewsModel.insertReview(data, function(err, dbResult) {
    if (err) {
      console.error('Error inserting review:', err);
      return res.status(500).json({ error: 'Internal server error' });
    }

    // 2) After inserting the review, call the Flask /predict endpoint
    // Map the numeric sessionDuration into the strings your AI expects
    const durationStr = mapMinutesToStr(data.sessionDuration);

    // Example: we pass schedule_satisfaction as "schedule_sat" to match your AI code
    axios.post('http://localhost:3000/predict', {
      typical_study_str: durationStr,
      break_freq_str: 'Sometimes',           // placeholder or from your UI
      schedule_sat: data.scheduleSatisfaction
    })
    .then(flaskResponse => {
      // 3) Return both the DB result and the AI prediction
      return res.status(201).json({
        message: 'Review created successfully',
        dbData: dbResult,                  // e.g., { review_id: ... }
        aiPrediction: flaskResponse.data   // e.g., { predicted_class, predicted_label, probabilities, etc. }
      });
    })
    .catch(aiError => {
      console.error('Error calling AI:', aiError);
      // Decide how you want to handle AI errors:
      // We'll still return 201 for the DB insertion, but note the AI call failed
      return res.status(201).json({
        message: 'Review created successfully (AI call failed)',
        dbData: dbResult
      });
    });
  });
};
