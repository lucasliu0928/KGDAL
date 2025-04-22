library(ontologyIndex)

get_ancestor_without_self <- function(ontology_df,term_id){
  ancestors_withSelf <- get_ancestors(ontology_df, term_id)
  ancestors <- ancestors_withSelf[-which(ancestors_withSelf == term_id)] #exclude curr id itself
  return(as.vector(ancestors))
}
#Dir
datadir <- "./Ontology_features/"
outdir <- "./Ontology_features/Commom_Ancestors/"

#Load HPO
ghpo_dat<- get_ontology(paste0(datadir,"hp2.obo.txt"),extract_tags = "everything")
get_relation_names(paste0(datadir,"hp2.obo.txt"))  #'@NOTE: HPO only have is_a relation

#1. All terms in HPO
all_termsIDs_inHPO <- ghpo_dat$id
all_termsNames_inHPO <- ghpo_dat$name

#All ID and names
HPO_IDandnames <- cbind.data.frame(as.vector(all_termsNames_inHPO),as.vector(all_termsIDs_inHPO))
colnames(HPO_IDandnames) <- c("Name","ID")

#Check childern of root 
childeren_IDs_of_root <- ghpo_dat$children[1]$`HP:0000001`
Childeren_Names_of_root <- as.vector(HPO_IDandnames[which(HPO_IDandnames[,"ID"] %in% childeren_IDs_of_root),"Name"])

#2. Get all descendants of Phenotypic abnormality
descendant_of_Pheno_abnorm <- get_descendants(ghpo_dat,"HP:0000118") #Phenotypic abnormality 
length(descendant_of_Pheno_abnorm) # 15560

#3. Get Ancestors of each term, and store ID, NAME and ancestors in one df
selected_terms_ids <- descendant_of_Pheno_abnorm #select terms to compute
selected_term_ancestor_df <- as.data.frame(matrix(NA,nrow = length(selected_terms_ids) ,ncol = 3))
colnames(selected_term_ancestor_df) <- c("ID","Name","Ancestors")
for (i in 1:length(selected_terms_ids)){
  if (i %% 100 == 0){
    print(i)
  }
  curr_id <- selected_terms_ids[i]
  curr_name <- as.character(HPO_IDandnames[which(HPO_IDandnames[,"ID"] == curr_id),"Name"])
  curr_ancestors_ids <- get_ancestor_without_self(ghpo_dat,curr_id) #get curr ancestors without selfv
  selected_term_ancestor_df[i,"ID"] <- curr_id
  selected_term_ancestor_df[i,"Name"] <- curr_name
  selected_term_ancestor_df[i,"Ancestors"] <- paste0(curr_ancestors_ids,collapse = "%")

}



#3. Compute common ancestors
n_randomSamples <- 100
common_ancestor_df_list <- list(NA)
for (i in 1:nrow(selected_term_ancestor_df)){
  if (i %% 100 == 0){
    print(i)
  }
  curr_id <- selected_term_ancestor_df[i,"ID"]
  curr_name <- selected_term_ancestor_df[i,"Name"]
  #get curr ancestors 
  curr_ancestors_ids <- unlist(strsplit(selected_term_ancestor_df[i,"Ancestors"],split = "%"))
  #random generate 100 (other terms)  + 1 (AKI)
  #1.random generate 100 without self and AKI
  AKI_idxes <- which(selected_term_ancestor_df[,"ID"] == "HP:0001919")
  self_idxes <- i
  all_indexes <- 1:nrow(selected_term_ancestor_df)
  all_available_withoutself_andAKI_indexes <- all_indexes[-c(AKI_idxes,self_idxes)]
  set.seed(i)
  pair_indxes <- sample(all_available_withoutself_andAKI_indexes,n_randomSamples, replace=FALSE)
  #2. Add AKI back, in this way , make sure we compute the common ancestors of each term with AKI
  updated_pair_indexes <- c(pair_indxes ,AKI_idxes)
  
  #3. get updated pair ID info in selected_term_ancestor_df
  pair_info <- selected_term_ancestor_df[updated_pair_indexes,]
  
  curr_common_ancestor_df <- as.data.frame(matrix(NA, nrow = nrow(pair_info),ncol = 5))
  colnames(curr_common_ancestor_df) <-c("ID1","ID2","Name1","Name2","N_Common_Ancestors")
  #Compute common ancestors between each curr term and pair IDs
  for (j in 1:nrow(pair_info)){
    #get pair info
    curr_pair_id <- pair_info[j,"ID"]
    curr_pair_name <- pair_info[j,"Name"]
    curr_pair_ancestors_ids <- unlist(strsplit(pair_info[j,"Ancestors"],split = "%"))
    
    #get common ancestors
    common_ancestors <- intersect(curr_ancestors_ids,curr_pair_ancestors_ids)
    n_common_ancestors <- length(common_ancestors)
    curr_common_ancestor_df[j,"ID1"] <- curr_id
    curr_common_ancestor_df[j,"ID2"] <- curr_pair_id
    curr_common_ancestor_df[j,"Name1"] <- curr_name
    curr_common_ancestor_df[j,"Name2"] <- curr_pair_name
    curr_common_ancestor_df[j,"N_Common_Ancestors"] <- n_common_ancestors
    
    #sort names and Ids, so that we can detect duplicates when names can swaped
    curr_common_ancestor_df[j,c("ID1","ID2")] <- sort(curr_common_ancestor_df[j,c("ID1","ID2")])
    curr_common_ancestor_df[j,c("Name1","Name2")] <- sort(curr_common_ancestor_df[j,c("Name1","Name2")])
    
  }
  
  common_ancestor_df_list[[i]] <- curr_common_ancestor_df
}


final_common_ancestor_df <- do.call(rbind,common_ancestor_df_list)
nrow(final_common_ancestor_df)

##remove duplicate relations, ID1=A, ID2=B; ID1=B,ID2=B
updated_common_ancestor_df<- final_common_ancestor_df[!duplicated(final_common_ancestor_df[,c("ID1","ID2","N_Common_Ancestors")]),]
nrow(updated_common_ancestor_df)
length(unique(updated_common_ancestor_df$N_Common_Ancestors)) #31
write.csv(updated_common_ancestor_df, paste0(outdir,"common_ancestors_df_Phenotypic_Abnormality.csv"))
