// reviewsModel.js
const { pool } = require('../services/db');

module.exports.insertReview = (data, callback) => {
    const SQLSTATEMENT = `
        INSERT INTO reviews (sessionDuration, rating, feedback, user_id, scheduleSatisfaction)
        VALUES (?, ?, ?, ?, ?)
    `;
    const VALUES = [
        data.sessionDuration,
        data.rating,
        data.feedback,
        data.user_id,
        data.scheduleSatisfaction
    ];

    pool.query(SQLSTATEMENT, VALUES, function (err, result) {
        if (err) return callback(err, null);

        // In MySQL, result.insertId is the auto-incremented primary key
        callback(null, { review_id: result.insertId });
    });
};
module.exports.selectAllReviews = (callback) => {
    const SQLSTATEMENT = `
        SELECT *
        FROM Reviews;
    `;
    const VALUES = [];
    pool.query(SQLSTATEMENT,VALUES, callback)
}