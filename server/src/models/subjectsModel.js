const { pool } = require('../services/db');

module.exports.selectAll = (callback) =>{
   
        const SQLSTATEMENT = `SELECT * FROM Subjects;`;
        pool.query(SQLSTATEMENT, [], callback);  // ✅ Add an empty array as params
}


module.exports.insertSubject = (data, callback) => {
    const SQLSTATEMENT = `
        INSERT INTO Subjects (name, description)
        VALUES (?, ?)
    `;
    const VALUES = [data.subject, data.description];
    
        pool.query(SQLSTATEMENT, VALUES, function (err, result) {
        if (err) return callback(err, null);
    
        callback(null, { subject_id: result.lastID });
    });
};
    