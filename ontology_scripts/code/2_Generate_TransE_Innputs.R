#DDIR
data_dir <- "./Ontology_features/Commom_Ancestors/"
outdir <- "./Ontology_features/TransE_Input/"
  
#0.Load Common ancestors
common_ancestors_df <- read.csv(paste0(data_dir,"common_ancestors_df_Phenotypic_Abnormality.csv"),stringsAsFactors = F)
common_ancestors_df <- common_ancestors_df[,-1]
#1.Reformat IDs
common_ancestors_df$ID1 <- gsub(":","",common_ancestors_df$ID1)
common_ancestors_df$ID2 <- gsub(":","",common_ancestors_df$ID2)

#2.Generate entity2 ids
unique_ids <- unique(c(common_ancestors_df[,"ID1"], common_ancestors_df[,"ID2"])) #15560
n_terms <- length(unique_ids)
embed_ids <- seq(0,n_terms-1,1)
hpo_ids_with_embed_id <- paste(unique_ids,embed_ids)
fileConn<-file(paste0(outdir,"entity2id.txt"))
writeLines(c(n_terms,hpo_ids_with_embed_id), fileConn)
close(fileConn)

#2.Generate relation2id.txt
unique_rel <- paste0("N_Common_Ancestors_", unique(common_ancestors_df$N_Common_Ancestors))
n_terms <- length(unique_rel)
rel_ids <- seq(0,n_terms-1,1)
rel_with_rel_id <- paste(unique_rel,rel_ids)
fileConn<-file(paste0(outdir,"relation2id.txt"))
writeLines(c(n_terms,rel_with_rel_id), fileConn)
close(fileConn)


#3.Generate train2id.txt
#A.Get train triplets
train_triplets <- common_ancestors_df[,c("ID1","ID2","N_Common_Ancestors")]
#B. update HPO ID with embed ID
train_triplets$EmbedID1 <- NA
train_triplets$EmbedID2 <- NA
for (i in 1:length(unique_ids)){
  if(i %% 100 == 0){
    print(i)
  }
  curr_id <- unique_ids[i]
  curr_idxes1 <- which(train_triplets[,"ID1"] == curr_id)
  curr_idxes2 <- which(train_triplets[,"ID2"] == curr_id)
  
  train_triplets[curr_idxes1,"EmbedID1"] <- embed_ids[i]
  train_triplets[curr_idxes2,"EmbedID2"] <- embed_ids[i]
}

n_triplets <- nrow(train_triplets)
train_triplets_comb <- paste(train_triplets[,"EmbedID1"],train_triplets[,"EmbedID2"],train_triplets[,"N_Common_Ancestors"])
fileConn<-file(paste0(outdir,"train2id.txt"))
writeLines(c(n_triplets,train_triplets_comb), fileConn)
close(fileConn)

#######
n_final_training_Sample <- (length(train_triplets_comb)*2)/3
train_triplets_comb2 <- sample(train_triplets_comb,n_final_training_Sample,replace = FALSE)
n_triplets <- length(train_triplets_comb2)
fileConn<-file(paste0(outdir,"train2id_V3.txt"))
writeLines(c(n_triplets,train_triplets_comb), fileConn)
close(fileConn)

#4.test2id.txt, and valid2id.txt
#'@NOTE: Just copy some from train2id,
#'Because we only need to get emebeddings from TransX models, we do not need to do any link prediction
test_triplets_comb <- sample(train_triplets_comb,500,replace = FALSE)
n_test <- length(test_triplets_comb)
fileConn<-file(paste0(outdir,"test2id.txt"))
writeLines(c(n_test,test_triplets_comb), fileConn)
close(fileConn)


valid_triplets_comb <- sample(train_triplets_comb,300,replace = FALSE)
n_valid <- length(valid_triplets_comb)
fileConn<-file(paste0(outdir,"valid2id.txt"))
writeLines(c(n_valid,valid_triplets_comb), fileConn)
close(fileConn)