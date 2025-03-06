// ##############################################################
// REQUIRE MODULES
// ##############################################################
const { pool } = require('../services/db');

// ##############################################################
// DEFINE INSERT NEW users
// ##############################################################
module.exports.insertUser = (data, callback) => {
    const SQLSTATEMENT = `
        INSERT INTO users (username, email, password)
        VALUES (?, ?, ?)
    `;
    const VALUES = [data.username, data.email, data.password];

    pool.query(SQLSTATEMENT, VALUES, function (err, result) {
        if (err) return callback(err, null);
        
        // Now result.lastID will contain the inserted user's ID
        callback(null, { user_id: result.lastID });
    });
};
// ##############################################################
// DEFINE MODEL FOR SELECT USER BY USERNAME
// ##############################################################
module.exports.selectUserByUsername = (data, callback) => {
    const SQLSTATEMENT = `
        SELECT *
        FROM users
        WHERE username = ?;
    `;

    const VALUES = [data.username];
    pool.query(SQLSTATEMENT, VALUES, callback)
}
// ##############################################################
// DEFINE MODEL FOR SELECT ALL USERS
// ##############################################################
module.exports.selectAllUsers = (callback) => {
    const SQLSTATEMENT = `
        SELECT *
        FROM users;
    `;
    const VALUES = [];
    pool.query(SQLSTATEMENT,VALUES, callback)
}
// ##############################################################
// DEFINE MODEL FOR UPDATING USER BY ID
// ##############################################################
module.exports.updateById = (data, callback) => {
    const SQL_UPDATE = `
        UPDATE users 
        SET username = ?
        WHERE user_id = ?;
    `;
    const VALUES = [data.username, data.user_id];

    // perform the update user by userid
    pool.query(SQL_UPDATE, VALUES,callback)
};
// ##############################################################
// DEFINE MODEL FOR SELECT USERS BY USER_ID
// ##############################################################
module.exports.selectUserByUserId = (data, callback) =>
    {
        const SQLSTATMENT = `
        SELECT * 
        FROM users
        WHERE user_id = ?;
        `;
        const VALUES = [data.user_id];
    
        pool.query(SQLSTATMENT, VALUES, callback);    
    }
//////////////////////////////////////////////////////
// SELECT USER BY USERNAME OR EMAIL
//////////////////////////////////////////////////////
module.exports.selectUserByUsernameOrEmail = (data, callback) =>
    {
        const SQLSTATMENT = `
        SELECT * FROM Users
        WHERE username = ? OR email = ?;
        `;
        const VALUES = [data.username, data.email];
    
        pool.query(SQLSTATMENT, VALUES, callback);    
    }

