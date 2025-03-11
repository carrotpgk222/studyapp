// ##############################################################
// REQUIRE MODULES
// ##############################################################
const model = require("../models/studyModel");

// ##############################################################
// CONTROLLER FUNCTION TO START A STUDY SESSION
// ##############################################################
module.exports.startStudySession = (req, res, next) => {
    if (!req.body.user_id || !req.body.subject_id) {
        res.status(400).send("Missing required data.");
        return;
    }

    const data = {
        user_id: req.body.user_id,
        subject_id: req.body.subject_id || null
    };

    const callback = (error, results) => {
        if (error) {
            console.error("Error startStudySession:", error);
            res.status(500).json(error);
        } else {
            res.status(201).json({
                message: "Study session started successfully",
                session_id: results.insertId,
                start_time: data.start_time
            });
        }
    };

    model.insertStudySession(data, callback);
};

// ##############################################################
// CONTROLLER FUNCTION TO END A STUDY SESSION
// ##############################################################
module.exports.endStudySession = (req, res, next) => {
    if (!req.body.session_id) {
        res.status(400).send("Missing required data.");
        return;
    }

    const data = {
        session_id: req.body.session_id
    };

    const callback = (error, results) => {
        if (error) {
            console.error("Error endStudySession:", error);
            res.status(500).json(error);
            return
        } else if (results.affectedRows === 0) {
            res.status(404).json({ message: "Study session not found." });
            return
        } else {
            console.log("Study session ended successfully")
            next()
        }
    };

    model.updateStudySessionEndTime(data, callback);
};

// ##############################################################
// DEFINE CONTROLLER FUNCTION TO CALCULATE TOTAL STUDY TIME
// ##############################################################
module.exports.calculateTotalStudyTime = (req, res, next) => {
    const data = {
        session_id: req.body.session_id
    };

    const callback = (error, results, fields) => {
        if (error) {
            console.error("Error calculateTotalStudyTime:", error);
            res.status(500).json(error);
        } else {
            if (results.length == 0) {
                return res.status(404).json({ message: "No study sessions found" });
            }

            // Calculate total time

            let totalTimeInMinutes = 0;
                const startTime = new Date(results.start_time);  // Assuming start_time is in 'YYYY-MM-DD HH:mm:ss' format
                const endTime = new Date(results.end_time);      // Assuming end_time is in 'YYYY-MM-DD HH:mm:ss' format
                
                const durationInMinutes = (endTime - startTime) / 60000;  // Convert milliseconds to minutes
                totalTimeInMinutes += durationInMinutes;
            
            res.locals.totalTime = totalTimeInMinutes
            
        } next()
    };

    model.selectStudySessionBySessionId(data, callback);
};


// ##############################################################
// DEFINE CONTROLLER FUNCTION TO DELETE A STUDY SESSION BY ID
// ##############################################################
module.exports.deleteStudySessionById = (req, res, next) => {
    const data = {
        session_id: req.params.session_id
    };

    const callback = (error, results, fields) => {
        if (error) {
            console.error("Error deleteStudySessionById:", error);
            res.status(500).json(error);
        } else if (results.affectedRows === 0) {
            res.status(404).json({ message: "Study session not found." });
        } else {
            res.status(200).json({ message: "Study session deleted successfully." });
        }
    };

    model.deleteStudySessionById(data, callback);
};

// ##############################################################
// CONTROLLER FUNCTION TO STORE THE END TIME
// ##############################################################
module.exports.storeEndTIme = (req, res, next) => {
    const data = {
        time: res.locals.totalTime
    };

    const callback = (error, results) => {
        if (error) {
            console.error("Error startStudySession:", error);
            res.status(500).json(error);
        } else {
            res.status(201).json({
                message: "Study session ended successfully"
            });
        }
    };

    model.insertTotalTime(data, callback);
};