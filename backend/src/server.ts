import express from 'express'
import dotenv from 'dotenv'
import {ChatOllama,OllamaEmbeddings} from '@langchain/ollama'
import {initChatModel,SystemMessage,HumanMessage} from 'langchain'
import {QdrantVectorStore} from '@langchain/qdrant'
import {Document} from '@langchain/core/documents'
import {StateGraph,START,END,Annotation,Command,MemorySaver} from '@langchain/langgraph'
import multer from 'multer'
import * as z from 'zod'

dotenv.config()
const app=express()
app.use(express.json())

const visionModel=new ChatOllama({
    model:'minicpm-v4.5:8b'
})
const model=await initChatModel('lfm2.5:8b',{
    modelProvider:'ollama'
})
const embeddingModel=new OllamaEmbeddings({
    model:'nomic-embed-text-v2-moe',
})
const checkpointer=new MemorySaver()
let thread_id=1

const dbUrl=process.env.QDRANT_URL
const dbApiKey=process.env.QDRANT_API_KEY
if(!dbUrl||!dbApiKey){
    throw new Error('QDRANT_URL and QDRANT_API_KEY are required')
}
const decisionVectorStore=await QdrantVectorStore.fromExistingCollection(embeddingModel,{url:dbUrl,apiKey:dbApiKey,collectionName:'traffic-decision'})
const solutionVectorStore=await QdrantVectorStore.fromExistingCollection(embeddingModel,{apiKey:dbApiKey,url:dbUrl,collectionName:'traffic-solution'})

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
const decisionModelSchema=z.object({
    isSolutionNeeded:z.boolean().describe(`
Determine whether the current traffic situation requires a traffic-management solution.

Consider:
- traffic density
- traffic order
- vehicle counts
- whether vehicles are simply queued at a traffic signal
- the scene observation
- the retrieved historical cases and their previous decisions

Do not assume that high vehicle count or a signal queue automatically means a solution is needed.
Use the historical cases as supporting evidence and decide based on how similar the current situation is to those cases.
The main factors are traffic order, density and observation.
`),
    reasoning:z.string().describe('Your reasoning behind this decision')
})

const decisionModelPrompt=`
You are a traffic decision agent.

You will receive:

1. The current traffic situation:
{
    vehicleCount: number,
    fourWheelerCount: number,
    twoWheelerCount: number,
    autoCount: number,
    trafficDensity: "low" | "medium" | "high",
    trafficOrder: "orderly" | "moderate" | "messy",
    observation: string
}

2. Historical cases that are similar to the current traffic situation.
Each historical case contains the traffic situation and the decision previously made for that case.

Determine whether the current traffic situation requires a solution.

Use the historical cases as supporting evidence, but do not blindly copy their decisions. Consider how closely they match the current situation.

Pay particular attention to whether apparent congestion is simply a normal, orderly traffic-signal queue rather than a genuine traffic problem.

Return only the requested structured output.
`

const solutionModelPrompt=`
You are a traffic-management solution agent.

The decision agent has already determined that the current traffic situation requires a solution.

You will receive:

1. The current traffic situation, including:
- vehicle counts
- traffic density
- traffic order
- scene observation

2. Historical cases containing:
- traffic situations similar to the current situation
- solutions that were previously proposed
- human ratings indicating how useful those solutions were (between 1 and 5)

Your task is to propose the most appropriate practical traffic-management solution for the current situation.

Use the historical cases as supporting evidence. Give greater consideration to solutions from highly similar cases that received better human ratings.

Do not blindly copy a historical solution. Adapt it to the current traffic situation.

The proposed solution should be:
- practical
- specific to the observed traffic problem
- feasible as a traffic-management intervention
- concise enough for a human reviewer to understand and evaluate

Do not reconsider whether a solution is needed. That decision has already been made by the decision agent.

Return only the proposed solution in plain text and no json response, only directly text response.
Do not reply in json with solution and human rating params
`
const decisionRagBody=z.object({
    input:visionModelSchema,
    output:decisionModelSchema
})
const solutionRagBody=z.object({
    input:visionModelSchema,
    output:z.object({
        solution:z.string(),
        humanRating:z.number().int().min(1).max(5)
    })
})

type SolutionRagAnnotation={
    input:z.infer<typeof visionModelSchema>,
    output:{
        solution:string,
        humanRating:number
    }
}

const graphAnnotation=Annotation.Root({
    visionModelResponse:Annotation<z.infer<typeof visionModelSchema>>(),
    decisionSimilaritySearchResponse:Annotation<z.infer<typeof decisionRagBody>[]>(),
    decisionModelResponse:Annotation<z.infer<typeof decisionModelSchema>>(),
    solutionSimilaritySearchResponse:Annotation<SolutionRagAnnotation[]>(),
    solutionModelResponse:Annotation<string>()
})

const graph=new StateGraph(graphAnnotation)
.addNode('decisionSimilaritySearch',async (state)=>{
    const similaritySearch=await decisionVectorStore.similaritySearch(nlFromStructuredLanguage(state.visionModelResponse),5)
    return {decisionSimilaritySearchResponse:similaritySearch.map(doc=>({input:doc.pageContent,output:JSON.stringify(doc.metadata)}))}
})
.addNode('decisionModel',async (state)=>{
    const res=await model.withStructuredOutput(decisionModelSchema).invoke([
        new SystemMessage(decisionModelPrompt),
        new HumanMessage(`
Historical cases: ${state.decisionSimilaritySearchResponse.map((data:any)=>`input:${data.input} output:${data.output}`).join('\n')}
Current Case: ${JSON.stringify(state.visionModelResponse)}
`)
    ])
    return new Command({goto:res.isSolutionNeeded?'solutionSimilaritySearch':END,update:{decisionModelResponse:res}})
},{
    ends:[END,'solutionSimilaritySearch']
})
.addNode('solutionSimilaritySearch',async (state)=>{
    const similaritySearch=await solutionVectorStore.similaritySearch(nlFromStructuredLanguage(state.visionModelResponse),5)
    return {solutionSimilaritySearchResponse:similaritySearch.map(doc=>({input:doc.pageContent,output:JSON.stringify(doc.metadata)}))}
})
.addNode('solutionModel',async (state)=>{
    const res=await model.withStructuredOutput(z.object({solution:z.string()})).invoke([
        new SystemMessage(solutionModelPrompt),
        new HumanMessage(`
Historical cases: ${state.solutionSimilaritySearchResponse.map(data=>`input:${data.input} output ${data.output}`).join('\n')}
Current case: ${JSON.stringify(state.visionModelResponse)}
`)
    ])
    return {solutionModelResponse:res.solution}
})
.addEdge(START,'decisionSimilaritySearch')
.addEdge('decisionSimilaritySearch','decisionModel')
.addEdge('solutionSimilaritySearch','solutionModel')
.addEdge('solutionModel',END)

const compiledGraph=graph.compile({checkpointer})


app.post('/image',upload.single('traffic_frame'),async (req,res)=>{
    if(!req.file){
        return res.status(400).json({error:'No file found'})
    }
    const buffer=req.file.buffer
    const stringBuffer=buffer.toString('base64')
    try{
        const structuredVisionModel=visionModel.withStructuredOutput(visionModelSchema)
        const visionModelResponse=await structuredVisionModel.invoke([
            {type:'system',content:visionModelSystemPrompt},
            {type:'human',content:[{type:'image_url',image_url:`data:${req.file.mimetype};base64,${stringBuffer}`}]}
        ])
        const currentThreadId=thread_id++
        const finalData=await compiledGraph.invoke({visionModelResponse},{configurable:{thread_id:currentThreadId}})
        if(!finalData.decisionModelResponse.isSolutionNeeded){
            return res.status(200).json({message:"No solution needed"})
        }
        return res.status(200).json({message:finalData.solutionModelResponse,thread_id:currentThreadId})
    }
    catch(err){
        console.log(err)
        return res.status(500).json({error:"Internal server error"})
    }
})

type DecisionRagType=z.infer<typeof decisionRagBody>

function documentFromStructuredLanguage(data:DecisionRagType){
    return new Document({
        pageContent:nlFromStructuredLanguage(data.input),
        metadata:data.output
    })
}
function nlFromStructuredLanguage(input:z.infer<typeof visionModelSchema>){
    return `
Traffic situation:
Vehicle count: ${input.vehicleCount}
Four-wheelers: ${input.fourWheelerCount}
Two-wheelers: ${input.twoWheelerCount}
Auto-rickshaws: ${input.autoCount}
Traffic density: ${input.trafficDensity}
Traffic order: ${input.trafficOrder}
Scene observation: ${input.observation}
`
}

app.post('/decisionRag',async (req,res)=>{
    const body=req.body
    const output=decisionRagBody.safeParse(body)
    if(!output.success){
        return res.status(400).json({error:output.error.issues.map(issue=>issue.message)})
    }
    const data=output.data
    const docs=documentFromStructuredLanguage(data)
    try{
        await decisionVectorStore.addDocuments([docs])
        return res.status(200).json({message:'Embedded sucessfully'})
    }catch(err){
        console.log(err)
        return res.status(500).json({error:"Internal server error"})
    }
})

app.post('/solutionRag',async (req,res)=>{
    const body=req.body
    const output=solutionRagBody.safeParse(body)
    if(!output.success){
        return res.status(400).json({error:output.error.issues.map(issue=>issue.message)})
    }
    const {data}=output
    const doc=new Document({
        pageContent:nlFromStructuredLanguage(data.input),
        metadata:{
            solution:data.output.solution,
            humanRating:data.output.humanRating
        }
    })
    try{
        await solutionVectorStore.addDocuments([doc])
        return res.status(200).json({message:"Embedded successfuly"})
    }catch(err){
        console.log(err)
        return res.status(500).json({error:"Internal server error"})
    }
})

const PORT=process.env.PORT||8000
app.listen(PORT,()=>console.log(`Server listening on port ${PORT}`))