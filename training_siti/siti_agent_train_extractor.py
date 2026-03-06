import agentlightning as agl
from typing import Any, Dict, List, Literal, Optional, cast
import argparse
from utils.logger import Logger
import pyarrow.parquet as pq
from models.ModelProviderClient import vLLMClient
from utils.Tool import ExtractorTool, PushTool
from utils.util import clean_response
from schema.structured_ouput_model import Tecnico
import re
import json
import torch
from time import sleep

logger = Logger(save=False, consoleLevel="INFO").getLogger()

CHAT_MODEL_SYSTEM_PROMPT= "Sei un assistente che estrae informazioni. Quando l'utente ti fornisce informazioni riguardanti le ore di lavoro fatte oppure le inefficienze trovate, DEVI chiamare la funzione extractor_expert per estrapolare le informazioni.\nDevi mandare alla funzione extractor_expert un riassunto di quello che l'utente ti ha detto riguardanti le ore di lavoro e le inefficienze trovate.\nDevi essere informale e non giudicare l'utente per le informazioni che lui ti da. Quando prendi in ingresso la risposta del extractor_expert non aggiungere informazioni sbagliate o pareri, devi ritornare quello che extractor_expert ritorna a te.\nDopo aver ricevuto la risposta dal extractor_expert chiedi conferma all'utente per salvare i dati. Se e solo se l'utente conferma le informazioni che sono state estratte te DEVI chiamare\nla funzione push_data per salvare le informazioni nel database, SOLO dopo l'utente ti a confermato che vanno bene, non prima  \nESEMPIO:\n[USER]: \"Ieri ho fatto 7 ore di lavoro\"\n[ASSISTENTE]: <chiama function extractor_expert con parametro: 'lavorato 7 ore '> \n[ASSISTENTE]: <extractor_expert ritorna -> {\n  \"ORE_ORDINARIE\": 7.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"INEFFICIENCY\": false,\n  \"NOTE\": null,\n  \"COMMESSA\": null,\n  \"risposta_singola\":\"\"\n} >\n[ASSISTENTE]: hai fatto 7 ore di lavoro senza inefficienze identificate, è corretto ?\n[USER]: si va bene \n[ASSISTENTE]: <chiama la function push_data>\n[ASSISTENTE]: <extractor_expert ritorna -> La commessa  è stata salvata >\n[ASSISTENTE]: la commessa è stata salvata sul database"
EXTRACTOR_MODEL_SYSTEM_PROMPT="Sei un assistente specializzato nell'estrazione di informazioni da messaggi per costruire una JSON.\n\nCOMPITO:\nAnalizza il messaggio dell'utente, identifica i dati rilevanti e costruisci un JSON con i campi specificati.\n\nREGOLE:\n- Estrai solo informazioni presenti esplicitamente nel testo\n- Se un campo richiesto non è presente nel messaggio, non inserire il suo valore\n- Non inventare o inferire dati non esplicitamente menzionati\n- Usa i valori esatti trovati nel testo\n- Costruisci il JSON nel formato richiesto\n\nCAMPI DA CERCARE:\ndevi cercare nel testo se si fa riferimento ai seguenti campi e salvali così come sono:\n- ORE_ORDINARIE: ore di lavoro ordinarie effettuate\n- ORE_STRAORDINARIE: ore di lavoro straorinario effettuate\n- ORE_VIAGGIO: ore di viaggio effettuate\n- DURATA_INEFFICIENCY: è un valore numerico che identifica quante ore a perso il tecnico per una inefficienza riscontrata a lavoro. se non trovi nessuna inefficienza settare questo campo a 0\n- NOTE_INEFFICIENCY: usalo per inserire la descrizione dell'inefficienza che il tecnico ha riscontrato. Se non trovi nessuna inefficienza deve essere vuoto. Deve essere breve e conciso.\n- NOTE: usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata. Altrimenti deve essere vuoto, deve essere breve e conciso senza l'utilizzo di tempi verbali. IMPORTANTE:la stringa deve essere sempre dentro a virgolette.\n- COMMESSA: id della commessa di lavoro, è una stringa numerica in questo formato 250376, può essere non sempre presente, se non la si trova settare a NULL il campo\n\nOUTPUT:\nRestituisci il JSON costruito con i dati estratti e i campi esattamente come ti ho passato, ritorna unicamente il JSON.\nOUTPUT ESEMPIO:\n{\n  \"ORE_ORDINARIE\": 0.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": null,\n  \"COMMESSA\": null\n}\nESEMPIO1:\n Ieri è stata una giornata di riposo\nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 0.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": riposo,\n  \"COMMESSA\": null\n}\nESEMPIO2:\n  Ieri per la commessa 290120 ho fatto 7 ore di lavoro\nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 7.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": null,\n  \"COMMESSA\": 290120\n}\nESEMPIO3:\n  Ieri ho lavorato 5 ore ma ho avuto un problema che mi è costato un ora del mio tempo: il cliente non mi forniva un muletto per svolgere un operazione \nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 5.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 1.0,\n  \"NOTE_INEFFICIENCY\":\"cliente non mi forniva un muletto per svolgere un operazione\",\n  \"NOTE\": null,\n  \"COMMESSA\": null\n}"
class SitiAgent:
    """
    Classe con all'interno la logica dell'agente di Siti 
    """
    def __init__(self, llm_resources:agl.NamedResources):
        # creiamo i due client 
        self.extractor_model: vLLMClient = vLLMClient(base_url=llm_resources.endpoint)
        
        # aggiungiamo i system prompt 
        self.extractor_model.add_system_prompt(EXTRACTOR_MODEL_SYSTEM_PROMPT)


    
    def chat(self, msg:str):
        """Funzione per scrivere un messaggio al chat model"""
        return self.chat_model.chat(message=msg, save_to_history=True, user="chat")

    def extract(self, msg:str)-> Tecnico|None:
        """Funzione per scrivere un messaggio al extractor model"""
        model_response = self.extractor_model.chat(message=msg, schema=Tecnico, user="extractor")
        # controlliamo che non sia avvenuto un errore:
        if "error" in model_response['status']:
            logger.error(f"errore nella estrazione dei dati: {model_response["status"]["error"]}")
            return None
        # puliamo da eventuali json tag non desiderati
        model_response = clean_response(model_response['content'])
        # castiamo e controlliamo i campi
        try: 
            json_response = json.loads(model_response)
            model_predict = Tecnico.model_validate_json(json.dumps(json_response))
        except Exception as e:
            logger.error({"error": f"Errore: {str(e)}\n model_response: {model_response}, ritorniamo None e continuiamo"})
            return None
        
        return model_predict

    def embed(self, prompt:str) -> list[float]:
        """Funzione per convertire un testo in embeddings"""
        result = self.embedding_model.embed(prompt=prompt, dim=256)
        logger.debug(f"embedding result for prompt: {prompt}: {result}")
        return result
    
    def save_tool_response(self,extractor_response:Any, tool_id:str, tool_name:str):
        """Funzione per salvare la risposta del tool nella history del chat model"""
        self.chat_model.save_2_conversation(text=extractor_response, role="tool", tool_name=tool_name, tool_call_id=tool_id)


        

class LitSitiExtractor(agl.LitAgent[Dict[str,Any]]):
    def __init__(self, *, trained_agents = None):
        super().__init__(trained_agents=trained_agents)


    def compute_embedding_reward(self, predict:str, gt:str)-> torch.Tensor:
        """Funzione per calcoalre il reward con gli embedding delle note""" 
        # se la gt è vuota utilizziamo un placeholder per valutare comunque se il predict è vicino 
        gt_embed =  torch.Tensor(self.agent.embed(gt)['embeddings'][0]['embedding']) if gt != None else  torch.Tensor(self.agent.embed("null")['embeddings'][0]['embedding'])
        predict_embed = torch.Tensor(self.agent.embed(predict)['embeddings'][0]['embedding']) if predict != None else  torch.Tensor(self.agent.embed("null")['embeddings'][0]['embedding'])
        # compute cosine similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        result = cos(predict_embed, gt_embed)

        return result

    def is_present(self,value:str)->bool:
        """funzione che trova se è presente una nota oppure no"""
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == '':
            return False
        return True

    def _compute_mean_reward(self, response:Tecnico, gt:dict)->float:
        denominator = 0
        accumulator = 0
        for key in gt.keys():
            match key:
                case 'ore_ordinarie':
                    if gt[key] != 0:
                        match = 1 if response.ore_ordinarie == gt[key] else 0
                        accumulator += match
                        denominator += 1
                    
                    elif response.ore_ordinarie != 0:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1 
                case 'ore_viaggio':
                    if gt[key] != 0:
                        match = 1 if response.ore_viaggio == gt[key] else 0
                        accumulator += match
                        denominator += 1
                    
                    elif response.ore_viaggio != 0:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1 
                case 'ore_straordinarie':
                    if gt[key] != 0:
                        match = 1 if response.ore_straordinarie == gt[key] else 0
                        accumulator += match
                        denominator += 1
                    
                    elif response.ore_straordinarie != 0:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1
                case 'durata_inefficienza':
                    if gt[key] != 0:
                        match = 1 if response.durata_inefficienza == gt[key] else 0
                        accumulator += match
                        denominator += 1
                    
                    elif response.durata_inefficienza != 0:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1
                case 'note':
                    gt_is_present = self.is_present(gt[key])
                    predict_is_present = self.is_present(response.note)
                    if gt_is_present != False:
                        match = 1 if predict_is_present == gt_is_present else 0
                        accumulator += match
                        denominator += 1
                    
                    elif predict_is_present != False:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1
                case 'note_inefficienza':
                    gt_is_present = self.is_present(gt[key])
                    predict_is_present = self.is_present(response.note_inefficienza) 
                    if  gt_is_present != False:
                        match = 1 if predict_is_present == gt_is_present else 0
                        accumulator += match
                        denominator += 1
                    
                    elif predict_is_present != False:
                        # falso positivo: true è 0 ma il modello ha predetto qualcosa
                        denominator += 1
                case _:
                    # la commessa è l'ultimo caso e non fa niente
                    continue
        
        if denominator == 0:
            # tutti i campi sono a zero sia in true che predict, si da il massimo
            return  1
        else:
            return accumulator / denominator


    def compute_extractor_reward(self, response:Tecnico,gt:dict, metric='all')-> float:
        """Funzione per calcolare i verifiable reward dell extractor model"""
        # e convertiamo in un json object
        
        vr_reward = 0
        # facciamo il confronto:
        match metric:
            case 'all': 
                # 1 se tutti i valori sono uguali, 0 altrimenti,

                # processiamo anche la note e le inefficienze delle note in modo simile, catturando solo il caso in cui la note è presente oppure no

                if response.ore_ordinarie == gt['ore_ordinarie'] and response.ore_straordinarie == gt['ore_straordinarie'] \
                    and response.ore_viaggio == gt['ore_viaggio'] and response.durata_inefficienza == gt['durata_inefficienza'] \
                        and self.is_present(response.note) == self.is_present(gt['note']) and self.is_present(response.note_inefficienza) == self.is_present(gt['note_inefficienza']):
                    vr_reward = 1.0
                else:
                    vr_reward = 0.0

            case 'mean':
                # fornire un reward come media dei risultati
                mean_score = self._compute_mean_reward(response,gt)

                # se tutti i  valori sono giusti passiamo un 1.5
                all_correct = (
                                response.ore_ordinarie == gt['ore_ordinarie'] 
                                and response.ore_straordinarie == gt['ore_straordinarie']
                                and response.ore_viaggio == gt['ore_viaggio'] 
                                and response.durata_inefficienza == gt['durata_inefficienza']
                                and self.is_present(response.note) == self.is_present(gt['note'])
                                and self.is_present(response.note_inefficienza) == self.is_present(gt['note_inefficienza'])
                )

                if all_correct:
                    vr_reward =  1.5
                elif mean_score > .5:
                    vr_reward = mean_score
                else:
                    vr_reward = 0
                            
                
        logger.info(f"vr_reward:{vr_reward}")
        return vr_reward
                
                

    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float|None:
        """
        Rollout function che fa arrivarre una task dallo scheduler di agent-lightning (nel nostro caso un elemnto proveniente dal dataset di training),
        una resorsa su cui fare il training (nel nostro caso llm endpoint di vLLM per modificare i pesi del modello) e dei metadati sulla rollout.
        Quello il nostro LitSitiAgent fa sará:
        1. prendere una task che contiene una conversazione completa dell'utente 
        2. per ogni user prompt si utilizza llm con le chiamate ai tools 
        3. si calcola il valore intermedio dei reward ad ogni step della conversazione 
        4. si emette un unico reward globale
        """
        llm: agl.LLM = cast(agl.LLM, resources['main_llm'])

        # init agent
        self.agent:SitiAgent = SitiAgent(llm)

        
        user_prompt = task["prompt"][0]["content"]
        gt = task['reward_model']['ground_truth']
        
        logger.info(f"user_prompt: {user_prompt}")
        logger.info(f"gt: {gt}")

        model_response = self.agent.extract(user_prompt)
        if model_response == None:
            logger.warning(f"estrattore andato in errore, mandiamo un reward nullo")
            return 0
        # calcoliamo il reward
        logger.info(f"model response: {model_response}")
        reward = self.compute_extractor_reward(model_response,gt=gt, metric='mean')
        logger.info(f"final reward: {reward}")
        return reward
    





def debug_siti_agent(args:Any):
    """
    funzione per fare il run dell'agente attraverso un llm locale prima di utilizzare VERL per controllare se il workflow avviene come atteso prima del RL
    """
    if args.train_file == None:
        logger.error("é necessario specificare un dataset di training per realizzare il debug")
        return
    table = pq.read_table(args.train_file)

    df = cast(List[Dict[str, Any]], table.to_pylist())
    logger.debug(f"Debug data: {df}")
    

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=args.vllm_endpoint,
                model="Qwen/Qwen3-4B-Instruct-2507",
                sampling_parameters={"temperature": 0.7}    
            )
        },
    )

    trainer.dev(LitSitiExtractor(), df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", action="store", dest="train_file", help="file path for parquet training data" 
    )
    parser.add_argument(
        "-e", "--endpoint", action="store", dest="vllm_endpoint", help="endpoint of vllm model" 
    )
    args = parser.parse_args()
    debug_siti_agent(args)