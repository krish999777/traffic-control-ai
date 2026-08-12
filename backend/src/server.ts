import express from 'express'
import dotenv from 'dotenv'
import {ChatOllama} from '@langchain/ollama'
import {initChatModel} from 'langchain'
import multer from 'multer'

const imageModel=new ChatOllama({
    model:'minicpm-v4.5:8b'
})

const model=initChatModel('lfm2.5:8b',{
    modelProvider:'ollama'
})

const upload=multer({
    storage:multer.memoryStorage(),
    limits:{
        fileSize:1024*1024*5
    },
    fileFilter:(req,file,cb)=>{
        if(file.mimetype.startsWith('image/')){
            cb(null,true)
        }else{
            cb(new Error('Only images are allowed'))
        }
    }
})

dotenv.config()
const app=express()

app.post('/image',upload.single('traffic_frame'),async (req,res)=>{
    if(!req.file){
        return res.status(400).json({error:'No file found'})
    }
    const buffer=req.file.buffer
    const stringBuffer=buffer.toString('base64')
    try{
        const visionModelResponse=await imageModel.invoke([
            {type:'system',content:`You will be given an image, you have to analyze this image and identify the scene, count the number of vehicles and classify them`},
            {type:'human',content:[{type:'image_url',image_url:`data:${req.file.mimetype};base64,${stringBuffer}`}]}
        ])
        return res.status(200).json({message:visionModelResponse.content})
    }
    catch(err){
        console.log(err)
        return res.status(500).json({error:"Internal server error"})
    }
})

const PORT=process.env.PORT||8000
app.listen(PORT,()=>console.log(`Server listening on port ${PORT}`))