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

app.post('/image',upload.single('traffic_frame'),(req,res)=>{
    if(!req.file){
        return res.status(400).json({error:'No file found'})
    }
    const buffer=req.file.buffer
})

const PORT=process.env.PORT||8000
app.listen(PORT,()=>console.log(`Server listening on port ${PORT}`))