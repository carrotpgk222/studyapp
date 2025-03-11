import React, { useState } from 'react';
import './review.css'; 
import { useNavigate } from 'react-router-dom';

const Review = () => {
  const navigate = useNavigate()
  const [selectedEmoji, setSelectedEmoji] = useState(null);
  const [selectedRating, setSelectedRating] = useState(null);
  const [feedback, setFeedback] = useState('');

  // Handle emoji selection
  const handleEmojiClick = (value) => {
    setSelectedEmoji(value); // e.g. 1, 2, 3, 4, 5
  };

  // Handle rating selection
  const handleRatingClick = (value) => {
    setSelectedRating(value); // e.g. 1, 2, 3, 4, 5
  };

  // Handle form submission
  const handleSubmitReview = (e) => {
    e.preventDefault();
    // Example: Hard-coded sessionDuration of 45 minutes
    // If you track the actual session length in state or props, use that instead
    const reviewData = {
      sessionDuration: 45,
      // This might store the "study session" rating as selectedEmoji
      rating: selectedEmoji,
      // This might store "AI satisfaction" as selectedRating
      scheduleSatisfaction: selectedRating,
      feedback: feedback,
      user_id: 1 // or wherever you get the logged-in user ID
    };

    // The endpoint URL for creating a new review
    const url = 'http://localhost:5000/api/reviews';

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      // The server expects fields named "username", "email", and "password"
      // If your server expects "name" instead of "username", adjust accordingly
      body: JSON.stringify({
        sessionDuration: 45,
      // This might store the "study session" rating as selectedEmoji
      rating: selectedEmoji,
      // This might store "AI satisfaction" as selectedRating
      scheduleSatisfaction: selectedRating,
      feedback: feedback,
      user_id: 1 // or wherever you get the logged-in user ID
      })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        // Registration successful
        console.log('Review successful:', data);
        alert('Review submitted successfully!');
        navigate("/profile")
      })
      .catch(error => {
        // Handle errors
        console.error('Review error:', error);
        alert('Review submission failed. Please try again.');
      });
  };

  return (
    <div className="body">
      <div className="logo">
        <img src="https://i.imgur.com/FLXaWSm.png" alt="Logo" />
      </div>

      <div className="container">
        <h1>Study Session Review</h1>
        <div className="timer">⏳ 45:00</div>
        <p>How was your study session?</p>

        {/* Emoji Rating */}
        <div className="emoji-rating">
          {['😡', '😞', '😐', '😊', '🤩'].map((emoji, index) => (
            <span
              key={index}
              className={`emoji ${selectedEmoji === index + 1 ? 'selected' : ''}`}
              onClick={() => handleEmojiClick(index + 1)}
            >
              {emoji}
            </span>
          ))}
        </div>

        {/* AI Suggestion Satisfaction */}
        <p>How satisfied are you with the AI suggestion?</p>
        <div className="rating-container">
          <div className="satisfaction-buttons">
            {[1, 2, 3, 4, 5].map((num) => (
              <div key={num} className="rating-item">
                {num === 1 && <span className="rating-label-above">Poor</span>}
                {num === 5 && <span className="rating-label-above">Excellent</span>}
                {num !== 1 && num !== 5 && <span className="rating-label-above empty-label"></span>}
                <button
                  className={`rating-btn ${selectedRating === num ? 'selected' : ''}`}
                  onClick={() => handleRatingClick(num)}
                >
                  {num}
                </button>
              </div>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmitReview}>
          <textarea
            className="feedback-box"
            placeholder="Optional feedback..."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
          />
          <button className="submit-btn" type="submit">
            Submit Review
          </button>
        </form>
      </div>
    </div>
  );
};

export default Review;
