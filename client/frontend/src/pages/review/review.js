import React, { useState } from 'react';
import './review.css'; // Import the CSS file

const Review = () => {
  const [selectedEmoji, setSelectedEmoji] = useState(null);

  // Handle emoji selection
  const handleEmojiClick = (value) => {
    setSelectedEmoji(value); // Set the selected emoji value
  };

  return (
    <div className="body">
      {/* Logo */}
      <div className="logo">
        <img
          src="https://i.imgur.com/FLXaWSm.png"
          alt="Logo"
        />
      </div>

      {/* Review Container */}
      <div className="container">
        <h1>Study Session Review</h1>

        {/* Larger Timer with More Space */}
        <div className="timer">⏳ 45:00</div>

        <p>How was your study session?</p>

        {/* Emoji Rating Section */}
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

        {/* Feedback Text Area */}
        <textarea className="feedback-box" placeholder="Optional feedback..."></textarea>

        {/* Submit Button */}
        <button className="submit-btn">Submit Review</button>
      </div>
    </div>
  );
};

export default Review;
