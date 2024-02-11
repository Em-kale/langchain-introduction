import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

console.log("starting the program");

console.log("initializing the chat model...");
//this is the chat model that we will use to generate the responses
const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "mistral",
});

//essentially sets what messages have happened alrady
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  ["user", "{input}"],
]);

async function loadVectorStore(): Promise<FaissStore> {
  console.log("loading the vector store...");
  const vectorStore = await FaissStore.load(
    "./vector_store/",
    new OllamaEmbeddings({ model: "mistral" })
  );
  console.log("finished loading the vector store.");
  return vectorStore;
}

async function doPrompt(){
  const loadedVectorStore = await loadVectorStore();

  console.log("`creating document chain...");

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  console.log("creating retriever...");
  //wraps the vector store so it conforms to the retriever interface
  //it in turn will use vector store methods to query the data , but lets you treat all retrievers the same
  const retriever = loadedVectorStore.asRetriever();

  console.log("creating retrieval chain...");
  //this should just handle adding the context to the input but I'm not confident how - need to investigate
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  console.log("invoking retrieval chain...");
  const result = await retrievalChain.invoke({
    input: "what is vital point ai",
  });
  console.log("finished.");
  console.log(result);
}

doPrompt();
