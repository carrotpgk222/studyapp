const { pool } = require("../services/db");

module.exports.insertStudySession = (data, callback) => {
    const SQLSTATEMENT = `
        INSERT INTO Study_Sessions (user_id, subject_id, start_time)
        VALUES (?, ?, DATETIME('now', 'localtime'))
    `;
    const VALUES = [data.user_id, data.subject_id];
    pool.query(SQLSTATEMENT, VALUES, (err, result) => {
        if (err) return callback(err, null);
        callback(null, { insertId: result.insertId });
    });
};

module.exports.updateStudySessionEndTime = (data, callback) => {
    const SQLSTATEMENT = `
        UPDATE Study_Sessions
        SET end_time = DATETIME('now', 'localtime')
        WHERE id = ?;
    `;
    pool.query(SQLSTATEMENT, [data.session_id], callback);
};

module.exports.selectStudySessionBySessionId = (data, callback) => {
    const query = "SELECT * FROM Study_Sessions WHERE id = ?";
    pool.query(query, [data.id], callback);
};

module.exports.insertTotalTime = (data, callback) => {
    const SQLSTATEMENT = `
        UPDATE Study_Sessions 
        SET total_time = ?
        WHERE id = ?;
    `;
    const VALUES = [data.time, data.session_id];
    pool.query(SQLSTATEMENT, VALUES, callback);
};
