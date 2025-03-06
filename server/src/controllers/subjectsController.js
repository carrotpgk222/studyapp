const model = require("../models/subjectsModel");


module.exports.getAllSubjects = (req, res, next) =>
    {
        const callback = (error, results, fields) => {
            if (error) {
                console.error("Error getAllSubjects:", error);
                res.status(500).json(error);
            } 
            else res.status(200).json(results);
        }
    
        model.selectAll(callback);
    }

module.exports.createNewSubject = (req, res, next) =>{
    const data = {
    subject: req.body.subject,
    description: req.body.description
     }
    if(req.body.subject == undefined)
        {
            res.status(400).send("Missing required data.");
            return;
        }
    const callback = (error, results, fields) => {
        console.log(results)
        if (error) {
            console.error("Error: createNewSubject", error);
            res.status(500).json(error);
        } else {
            res.status(201).json({
                message: "Subject successfully created",
                subject_id: results.subject_id,  
                subject: data.subject,   
                description: data.description        
            });
        };
    }
    
    model.insertSubject(data, callback);
}