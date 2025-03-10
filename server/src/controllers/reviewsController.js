// reviewsController.js
const reviewsModel = require('../models/reviewsModel');
const axios = require('axios');

/**
 * Convert a numeric duration (e.g. 45) into the strings your AI expects:
 *  - <30 -> "Less than 30 minutes"
 *  - <60 -> "30-60 minutes"
 *  - <120 -> "1-2 hours"
 *  - >=120 -> "More than 2 hours"
 */
function mapMinutesToStr(duration) {
  if (duration < 30) return 'Less than 30 minutes';
  if (duration < 60) return '30-60 minutes';
  if (duration < 120) return '1-2 hours';
  return 'More than 2 hours';
}

module.exports.createReview = (req, res) => {
  // Extract data from the request body
  const data = {
    sessionDuration: req.body.sessionDuration,       // e.g., 45
    rating: req.body.rating,                         // e.g., 1..5 for emojis
    feedback: req.body.feedback || '',               // optional text
    user_id: req.body.user_id,                       // user ID from the client
    scheduleSatisfaction: req.body.scheduleSatisfaction // e.g., 1..5 for AI satisfaction
  };

  // Validate required fields
  if (
    data.sessionDuration == null ||
    data.rating == null ||
    data.scheduleSatisfaction == null ||
    data.user_id == null
  ) {
    return res.status(400).json({ error: 'Missing required fields.' });
  }

  // 1) Insert the review into your DB via the model
  reviewsModel.insertReview(data, function(err, dbResult) {
    if (err) {
      console.error('Error inserting review:', err);
      return res.status(500).json({ error: 'Internal server error' });
    }

    // 2) After inserting the review, call the Flask /predict endpoint
    //    Convert the numeric duration into a string your AI model expects
    const durationStr = mapMinutesToStr(data.sessionDuration);

    axios.post('http://localhost:3000/predict', {
      // typical_study_str is used by your AI to encode study intensity
      typical_study_str: durationStr,

      // Hard-coded break_freq_str, or you could let the user choose
      break_freq_str: 'Sometimes',

      // schedule_sat is also used by your AI
      schedule_sat: data.scheduleSatisfaction
    })
    .then(flaskResponse => {
      // 3) Return both the DB result and the AI prediction to the client
      return res.status(201).json({
        message: 'Review created successfully',
        dbData: dbResult,                // e.g., { review_id: 123 }
        aiPrediction: flaskResponse.data // e.g., { predicted_class, predicted_label, ... }
      });
    })
    .catch(aiError => {
      console.error('Error calling AI:', aiError);
      // We still return 201 for the DB insertion but note the AI call failed
      return res.status(201).json({
        message: 'Review created successfully (AI call failed)',
        dbData: dbResult
      });
    });
  });
};
