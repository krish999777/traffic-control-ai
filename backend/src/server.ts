import express from 'express'
import dotenv from 'dotenv'
import {ChatOllama} from '@langchain/ollama'
import {initChatModel} from 'langchain'
import multer from 'multer'
import * as z from 'zod'

dotenv.config()
const app=express()

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

const visionModelSchema=z.object({
    vehicleCount:z.number().describe("The number of vehicles in the scene, including all the vehicles, 4 wheelers, 2 wheelers and auto rikshaws. Make sure the vehicleCount adds up to atleast the sum of fourWheelerCount,twoWheelerCount,autoCount(it can be more if there are other uncategorized)"),
    fourWheelerCount:z.number().describe("The number of 4 wheelers in the scene (cars,trucks etc)"),
    twoWheelerCount:z.number().describe("The number of 2 wheelers in the scene (scooters,bikes)"),
    autoCount:z.number().describe("The number of auto rikshaws in the scene (also known as tuk tuk, auto, rikshaws)"),
    trafficDensity:z.enum(['low','medium','high']).describe(`
The density of traffic put into scale. Rate it accordingly between low, medium, high.
If there is none or very minimal keep it in low itself.
Make sure you dont choose high only if there is too many vehicles.
If vehicles are in queue/signal in orderly manner then don't choose high.
A large number of vehicles waiting at a traffic signal does not automatically mean high traffic density. If vehicles are orderly and the queue is a normal signal-controlled queue with reasonable spacing, classify based on the actual physical density rather than the queue length.
`),
    trafficOrder:z.enum(['orderly','moderate','messy']).describe(`
orderly if vehicles are moving in queue/orderly manner.
moderate if there is some disorder or irregular movement.
messy if significant disorder, crowding, blocking, irregular movement, etc.
Do not choose messy if vehicles are following a queue/signal.
`),
    observation:z.string().describe(`
A short summary of the scene to mention things not include in the fields above namely vehicleCount, fourWheelerCount, twoWheelerCount, autoCount, trafficDensity, trafficOrder
I want you to specifically mention here if the vehicles are queued for a traffic signal. Also if vehicles are in messy state due to being queued in traffic signals then mention so clearly.
`)
})
const visionModelSystemPrompt=`
You will be provided an image for a traffic intersection. You have to respond in the structred output schema as provided
Remember to be accurate on all the reading and take your time
`

app.post('/image',upload.single('traffic_frame'),async (req,res)=>{
    if(!req.file){
        return res.status(400).json({error:'No file found'})
    }
    const buffer=req.file.buffer
    const stringBuffer=buffer.toString('base64')
    try{
        const structuredVisionModel=imageModel.withStructuredOutput(visionModelSchema)
        const visionModelResponse=await structuredVisionModel.invoke([
            {type:'system',content:visionModelSystemPrompt},
            {type:'human',content:[{type:'image_url',image_url:`data:${req.file.mimetype};base64,${stringBuffer}`}]}
        ])
        return res.status(200).json({message:visionModelResponse})
    }
    catch(err){
        console.log(err)
        return res.status(500).json({error:"Internal server error"})
    }
})

const PORT=process.env.PORT||8000
app.listen(PORT,()=>console.log(`Server listening on port ${PORT}`))