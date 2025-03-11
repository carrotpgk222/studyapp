// reviewsController.js
const reviewsModel = require('../models/reviewsModel');
const axios = require('axios');

/** 
 * 1) Create Review in DB (no AI call)
 */
module.exports.createReview = (req, res) => {
  const data = {
    sessionDuration: req.body.sessionDuration,
    rating: req.body.rating,
    feedback: req.body.feedback || '',
    user_id: req.body.user_id,
    scheduleSatisfaction: req.body.scheduleSatisfaction
  };

  if (
    data.sessionDuration == null ||
    data.rating == null ||
    data.scheduleSatisfaction == null ||
    data.user_id == null
  ) {
    return res.status(400).json({ error: 'Missing required fields.' });
  }

  reviewsModel.insertReview(data, (err, dbResult) => {
    if (err) {
      console.error('Error inserting review:', err);
      return res.status(500).json({ error: 'Internal server error' });
    }

    // Return a success response with the DB result
    return res.status(201).json({
      message: 'Review created successfully',
      dbData: dbResult
    });
  });
};

/**
 * 2) Separate Endpoint to Call AI
 * 
 * Expects request body with:
 *  - sessionDuration (number)
 *  - breakTime (number)
 *  - scheduleSatisfaction (number)
 */
module.exports.callAI = (req, res) => {
  const { review_id, sessionDuration, breakTime, scheduleSatisfaction } = req.body;

  // Validate input
  if (
    review_id == null ||
    sessionDuration == null ||
    breakTime == null ||
    scheduleSatisfaction == null
  ) {
    return res.status(400).json({ error: 'Missing required fields for AI call.' });
  }

  // Debug: log data being sent to AI
  console.log('Sending to AI:', {
    study_time: sessionDuration,
    break_time: breakTime,
    schedule_satisfaction: scheduleSatisfaction
  });

  // Call Flask /predict
  axios.post('http://127.0.0.1:5000/predict', {
    study_time: sessionDuration,
    break_time: breakTime,
    schedule_satisfaction: scheduleSatisfaction
  })
    .then(flaskResponse => {
      const aiData = flaskResponse.data; // e.g. { predicted_class, predicted_duration, probabilities }

      // Now store the AI result in time_prediction table
      const insertData = {
        review_id: review_id,
        predicted_class: aiData.predicted_class,
        predicted_duration: aiData.predicted_duration,
        probabilities: JSON.stringify(aiData.probabilities)
      };

      reviewsModel.insertTimePrediction(insertData, (err, result) => {
        if (err) {
          console.error('Error inserting AI prediction:', err);
          return res.status(500).json({ error: 'Error storing AI prediction' });
        }

        // Return success with AI data
        return res.status(200).json({
          message: 'AI prediction successful, stored in DB',
          aiPrediction: aiData
        });
      });
    })
    .catch(aiError => {
      console.error('Error calling AI:', aiError.message);
      if (aiError.response) {
        console.error('AI response status:', aiError.response.status);
        console.error('AI response data:', aiError.response.data);
      } else {
        console.error('No response received from AI');
      }

      return res.status(500).json({
        error: 'Failed to call AI',
        details: aiError.message
      });
    });
};


module.exports.getAllReviews = (req, res, next) =>{
  const callback = (error, results, fields) => {
      if (error) {
          console.error("Error getAllReviews:", error);
          res.status(500).json(error);
      } 
      else res.status(200).json(results);
  }

  reviewsModel.selectAllReviews(callback);
}
