import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "mistral",
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a newspaper reported from the early twentieth century"],
  ["user", "{input}"],
]);
const chain = prompt.pipe(chatModel);

async function doPrompt() {
  let result = await chain.invoke({ input: "what is langchain?" });
  console.log(result.content);
  //TODO: convert this output to a string instead of a message
}

doPrompt();
