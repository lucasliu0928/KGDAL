library("rjson")
change_name_func <- function(term_embed_df,name_col,old_name, new_name) {
  idx <- which(term_embed_df[,name_col]== old_name)
  if (length(idx) > 0 ){
    term_embed_df[idx,name_col] <- paste0(new_name,"_",term_embed_df[idx,name_col])
  }
  return(term_embed_df)
}


data_dir <- "./Ontology_features/"

#Read embeddings from using 50000 triplets to train (V2)
embedding_output <- fromJSON(paste(readLines(paste0(data_dir,"TransE_Output/V2embedding.vec.json")), collapse=""))
ent_embedding <- embedding_output$ent_embeddings.weight
rel_embedding <- embedding_output$rel_embeddings.weight

#Get entity embedding ID file 
entity_ID_df <- read.table(paste0(data_dir,"TransE_Input/entity2id.txt"),skip = 1)
colnames(entity_ID_df) <- c("HPO_ID","Embed_ID")

#Get relation embedding ID file 
relation_ID_df <- read.table(paste0(data_dir,"TransE_Input/relation2id.txt"),skip = 1)
colnames(relation_ID_df) <- c("Relation_ID","Embed_ID")

#Get common ancestor data
common_ancestors_df <- read.csv(paste0(data_dir,"Commom_Ancestors/common_ancestors_df_Phenotypic_Abnormality.csv"),stringsAsFactors = F)
common_ancestors_df$ID1 <- gsub(":","",common_ancestors_df$ID1)
common_ancestors_df$ID2 <- gsub(":","",common_ancestors_df$ID2)


#Terms for our model
selected_terms <- c("HP:0001919", #AKI
                    "HP:0004421","HP:0500105",  #SBP:  "Elevated systolic blood pressure","Decreased systolic blood pressure"
                    "HP:0005117","HP:0500104",  #DBP:  "Elevated diastolic blood pressure", Decreased diastolic blood pressure"
                    "HP:0012101",               #Scr:  "Decreased serum creatinine"
                    "HP:0032065",               #Bicarbonate : "Abnormal serum bicarbonate concentration"
                    "HP:0001899", "HP:0031851", #Hematocrit:   "Increased hematocrit", "Reduced hematocrit"
                    "HP:0002153", "HP:0002900", #Potassium:     "Elevated serum potassium levels", "Low blood potassium levels"
                    "HP:0002904", "HP:0033480",  #Billirubin: "High blood bilirubin levels",  "Decreased circulation of bilirubin in the blood circulation"
                    "HP:0003228", "HP:0002902",  #Sodium:      "High blood sodium levels", "Low blood sodium levels"
                    "HP:0001945", "HP:0002045",  #Temp:        "fewer", "Abnormally low body temperature"
                    "HP:0001974", "HP:0001882",  #WBC:         "High white blood count", "Low white blood cell count"
                    "HP:0031860",  #HR: "Abnormal heart rate variability"
                    "HP:0002793")  #RR "Abnormal pattern of respiration"

#Get embedding ID
selected_term_df <- entity_ID_df[which(entity_ID_df[,"HPO_ID"] %in% gsub(":","",selected_terms)),]
selected_term_embed_IDs_indexes <- selected_term_df$Embed_ID + 1 #plus one here, because ID 0 = idx 1 in ent_embedding list
selected_term_embed_vector <- ent_embedding[selected_term_embed_IDs_indexes]
selected_term_embed_df <- as.data.frame(do.call(rbind,selected_term_embed_vector))
rownames(selected_term_embed_df) <- selected_term_df$HPO_ID

#reorder
reorder_idexes <- match(gsub(":","",selected_terms),rownames(selected_term_embed_df))
selected_term_embed_df <- selected_term_embed_df[reorder_idexes,]

#Match names and Ids with embed vector df
#Load HPO
library(ontologyIndex)
ghpo_dat<- get_ontology(paste0(datadir,"hp2.obo.txt"),extract_tags = "everything")
#'@NOTE: HPO only have is_a relation
get_relation_names(paste0(datadir,"hp2.obo.txt")) 

#1. All terms in HPO
all_termsIDs_inHPO <- ghpo_dat$id
all_termsNames_inHPO <- ghpo_dat$name

#All ID and names
HPO_IDandnames <- cbind.data.frame(as.vector(all_termsNames_inHPO),as.vector(all_termsIDs_inHPO))
colnames(HPO_IDandnames) <- c("Name","ID")

#get names of HPO terms
selected_name_df <- HPO_IDandnames[which(HPO_IDandnames$ID %in% selected_terms),]
updated_selected_name_df <- selected_name_df[match(rownames(selected_term_embed_df),gsub(":","",selected_name_df$ID)),]
selected_term_embed_df$HPO_name <- as.character(updated_selected_name_df$Name)

#reorder_column
selected_term_embed_df <- selected_term_embed_df[,c(201,1:200)]

#rename some of the terms
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hyperkalemia","High_Potassium")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hypokalemia","Low_Potassium")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hyperbilirubinemia","High_Bilirubin")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hypobilirubinemia","Low_Bilirubin")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hypernatremia","High_Sodium")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hyponatremia","Low_Sodium")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Fever","High_Temp")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Hypothermia","Low_Temp")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Leukocytosis","High_WBC")
selected_term_embed_df <- change_name_func(selected_term_embed_df, "HPO_name","Leukopenia","Low_WBC")

write.csv(selected_term_embed_df,paste0(data_dir, "ent_embeddings.csv"),row.names = F)

#Todo
#Change name in common_ancestors_df
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hyperkalemia","High_Potassium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hypokalemia","Low_Potassium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hyperbilirubinemia","High_Bilirubin")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hypobilirubinemia","Low_Bilirubin")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hypernatremia","High_Sodium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hyponatremia","Low_Sodium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Fever","High_Temp")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Hypothermia","Low_Temp")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Leukocytosis","High_WBC")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name1","Leukopenia","Low_WBC")

common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hyperkalemia","High_Potassium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hypokalemia","Low_Potassium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hyperbilirubinemia","High_Bilirubin")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hypobilirubinemia","Low_Bilirubin")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hypernatremia","High_Sodium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hyponatremia","Low_Sodium")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Fever","High_Temp")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Hypothermia","Low_Temp")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Leukocytosis","High_WBC")
common_ancestors_df <- change_name_func(common_ancestors_df, "Name2","Leukopenia","Low_WBC")

#Get relation emebeddings
all_names <- selected_term_embed_df$HPO_name[-1]
relation_embeddings_df <- as.data.frame(matrix(NA, nrow = length(all_names), ncol = 202))
colnames(relation_embeddings_df) <- c("Relation_Terms","N_Common_Ancestors",paste0("V",seq(1,200)))
for (i in 1:length(all_names)){
  curr_name <- all_names[i]
  curr_idx <- which( (common_ancestors_df$Name1 == curr_name & common_ancestors_df$Name2 == "Acute kidney injury") |
               (common_ancestors_df$Name2 == curr_name & common_ancestors_df$Name1 == "Acute kidney injury") )
  curr_n_ca <- common_ancestors_df[curr_idx,"N_Common_Ancestors"]
  curr_relation_terms <- paste0("Acutekidneyinjury", "_to_",gsub(" |_","",curr_name))
  
  relation_embeddings_df[i,"Relation_Terms"] <- curr_relation_terms
  relation_embeddings_df[i,"N_Common_Ancestors"] <- curr_n_ca
  
  curr_relation_embedding_ID <- relation_ID_df[which(relation_ID_df$Relation_ID == paste0("N_CommonAncestors_",curr_n_ca)),"Embed_ID"]
  curr_relation_emebdding_index <-  curr_relation_embedding_ID + 1 #plus one here, because ID 0 = idx 1 in rel_embedding list
  curre_rel_embedding <- rel_embedding[[curr_relation_emebdding_index]]
  relation_embeddings_df[i,3:202] <- curre_rel_embedding
}

write.csv(relation_embeddings_df,paste0(data_dir, "rel_embeddings.csv"),row.names = F)
