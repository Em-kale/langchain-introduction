import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

async function generateVectorStore(): Promise<void> {
  //in memory vector store that works in a node environment
  console.log("starting the program");
  console.log("loading the documents from the vitalpoint.ai website...");

  const loader = new CheerioWebBaseLoader(
    //load in information about langchain
    "https://vitalpoint.ai/"
  );
  const docs = await loader.load();

  console.log("finished loading the documents from the vitalpoint.ai website");

  //split the text into more manageable chunks
  console.log("splitting the documents into smaller chunks...");
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  console.log("creating the vector store...");

  //this will take our embeddings model, which uses mixtral embeddings, and apply that to the splitDocs, generating
  //the relationships of the tokens in vector space
  const vectorStore = await FaissStore.fromDocuments(
    splitDocs,
    new OllamaEmbeddings({ model: "mistral" })
  );

  // Save the vector store to a directory
  const directory = "./vector_store/";
  await vectorStore.save(directory);
}

generateVectorStore();
