from __future__ import annotations
from dataclasses import dataclass 
from typing import Optional, Literal, List 
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from nlpaug.flow import Sequential 

AugmentMode = Literal["eda", "none"]

@dataclass
class AugmentConfig:
    """
    Configuration for text Augmentation.

    - eda : applies lightweight EDA-style augmentation (synonyms + small noise)
    - none: returns the input unchanged

    n_aug_per_sample:
        How many augmented versions to create per original example.
        If n_aug_per_example = 1 doubles the training size, original + 1 aug

    seed:
        Random seed for reproductability.
    
    min_tokens:
        Skip augmentation if the text is too short
    
    keep_original:
        Whether to keep original samples in the returned dataframe.
    """
    mode: AugmentMode = "eda"
    n_aug_per_sample: int = 1 
    seed: int = 42
    min_tokens: int = 4
    keep_original: bool = True 

class TextAugmenter:
        """
        A wrapper arround nplaug to generate augmented text samples.
        Designed for train-only augmentation (not on test/val)
        """

        def __init__(self, config: AugmentConfig):
            self.cfg = config
            self._rng = np.random.default_rng(self.cfg.seed)

            self._eda_pipeline = None
            if self.cfg.mode == "eda":
                self._eda_pipeline = self._build_eda_pipeline()

        def _build_eda_pipeline(self):
            """
            Builds a simple EDA-like pipeline using nlpaug.
            We use WordNet synonyms + small random noise operations.
            """

            STOPWORDS = {"it", "i", "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "are", "was", "were"}

            aug_syn = naw.SynonymAug(
            aug_src="wordnet",
            stopwords=list(STOPWORDS),
            aug_p=0.1,
            aug_max=2
            )           

            

            # Rabdin swap (small)
            aug_swap = naw.RandomWordAug(action='swap', aug_p=0.02)

            

            # Order is crucial: synonyms first then light noise
            return Sequential([aug_syn, aug_swap])

        def augment_text(self, text:str) -> List[str]:
            """
            Retruns a list of augmented texts (length = n_aug_per_sample).
            If the sample is too short returns an empty list.
            """    
            if self.cfg.mode == "none" or self.cfg.n_aug_per_sample <= 0: 
                return []
                
            tokens = text.split()

            if len(tokens) < self.cfg.min_tokens:
                return []
                
            if self.cfg.mode == "eda":
                # nlpaug internally uses randomness; we also shuffle a little to reduce duplicates
                augmented = self._eda_pipeline.augment(text, n=self.cfg.n_aug_per_sample)

                # nlpaug can return a single string if n=1, normalize to list
                # nlpaug's augment() method has inconsistent return types depending on n and the augmenter
                # augment(text, n=1) -> possible return type str or Listr[str]
                # augment(text, n>1) -> List[str]
                
                if isinstance(augmented, str):
                     augmented = [augmented]

                # Sometimes augmentation returns the original unchanged; deduplicate lightly
                # Example 
                # text = "I enjoy deep learning"
                # augmented = [
                #     " I enjoy deep learning ",
                #     "I enjoy profound learning",
                #     "",
                #     None
                # ]
                # output ["I enjoy profound learning"]

                augmented = [a.strip() for a in augmented if isinstance(a, str) and a.strip()]
                augmented = [a for a in augmented if a != text]

                # Return up to n_aug_per_sample clean augmentations.
                # if filtering reduced the count, keep fewer rather than forcing low-quality or duplicate samples.
                return augmented[: self.cfg.n_aug_per_sample]
                
            # Gurantees a stable return type, a list in our case
            return [] 
    
def augment_dataframe(
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        # Can be AugmentConfig object or None
        config: Optional[AugmentConfig] = None,       
    ) -> pd.DataFrame:
        
        """
        Augments a dataframe by creating additional rows with augmented text.

        TRAIN - only.
        Labels are preserved.
        Returns a new DF (does not modify input).

        Parameters:
        df : pd.DataFrame
            Must contain 'text_col' and 'label_col'.
        text_col : str
            Name of the text column.
        label_col : str 
            Name of the label column.
        config : AugmentConfig
            Augmentation settings.

        Returns
        a pd.DataFrame
            Augmented dataframe with columns: [text_col, label_col].
        """

        if config is None:
            config = AugmentConfig() 

        augmenter = TextAugmenter(config)

        # This will eventually become our new augmented DF
        # each element in this list will be 
        # {
        #   "text": "...augmented...",
        #   "label": "...same label...",
        #   "other_col": "...same metadata..."
        # }
        # structure List[Dict[str, Any]]
        augmented_rows = []

        TEXT_COL = text_col 
        LABEL_COL = label_col
        

        # Columns we will keep if present in df
        keep_cols = [c for c in df.columns if c != TEXT_COL]

        final_cols = [TEXT_COL] + keep_cols 
        

        # Loop over the original df
        for _, row in df.iterrows():
            # Takes the original text, returns a list of augmented strings
            aug_texts = augmenter.augment_text(row[TEXT_COL])

            # Loop over each augment text
            for aug_text in aug_texts:
                new_row = {}

                # This replaces the original text with the augmented version
                new_row[TEXT_COL] = aug_text
                # Augmentation must not change labels
                new_row[LABEL_COL] = row[LABEL_COL]

                for col in keep_cols:
                    # Skip, already included
                    if col != LABEL_COL:
                        # metadata preservation
                        new_row[col] = row[col]
                # Take each newly constructed augmented example
                # and add it as one new row to the list that will become the augmented DF
                augmented_rows.append(new_row)
        
        # Final Augmented DF
        aug_df = pd.DataFrame(augmented_rows, columns=final_cols) 

        # Selects only the columns we want from the original df
        # copy so nothing modifies the original df  
        original_subset = df[final_cols].copy()

        # Design switch
        if config.keep_original:
             # stack original rows on top of augmented rows
             # result: original samples + augmented samples
             output = pd.concat([original_subset, aug_df], ignore_index=True)
        else:
             # Drop original samples keep only augmented
             output = aug_df.reset_index(drop=True)
        # sample 100% of rows, shuffle the dataframe,  reproductability
        output = output.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
        return output 


def class_balanced_augment(
          df: pd.DataFrame,
          text_col: str ='text',
          label_col: str ='label',
          target_per_class: Optional[int] = None,
          config: Optional[AugmentConfig] = None,
) -> pd.DataFrame:
        # Original Dataset
        # + extra augmented samples(minority classes only)
        # Augment minority classes so each class reaches `target_per_class
        # Balanced Training Dataset

        if config is None:
             config = AugmentConfig()

        augmenter = TextAugmenter(config)

        # if target_per_class not specified, balance to the largest class
        if target_per_class is None:
             target_per_class = int(df[label_col].value_counts().max())

        augmented_rows = []
        base_cols = list(df.columns)
        rng = np.random.default_rng(config.seed)

        # Group data by class
        # new smaller dataframes for each label 
        for label, group in df.groupby(label_col):
             
            # how many real examples exist for this label
            current_n = len(group)
            # computes how many new augmented samples to add for this class.
            # if target is 1000 and class has 200 we need 800 more
            # max because some classes might already be at or above that.
            needed = max(0, target_per_class - current_n)

            if needed == 0:
                continue # already balanced
            
            # Converts the class-specific dataframe into a list of dictionaries, one per row
            # orient='records' because with this parameter we create a list
            # [{column -> value}, ... , {column -> value}]
            group_rows = group.to_dict(orient='records')

            # Counter: how many times we tried to produce an augmented sample
            # Augmentation can fail due to restrictions, too short text, WordNet synonym not found etc
            attemps = 0
            # safety cap
            # infinite loop can happen, for example in short texts (not in dair-ai emotion)
            # augmenter refuses to augment because min_tokens = 4 , it must stop trying so we set a cap
            max_attempts = needed * 5

            while(
                 #list comprehension that goes through all augmented rows 
                 #keeps only those belonging to the current class
                 # len() < needed counts how many augmenter rows we already created for this class
                 # keep looping until we reach target
                 len([r for r in augmented_rows if r[label_col] == label]) < needed
                 # Do not exceed max attempts
                 and attemps < max_attempts 
             ):
                 # loop increment regardless of success.
                 attemps += 1

                 # sample a real example from this class
                 # Controlled randomness due training
                 # augmentation randomness is isolated
                 # same seed -> same augmented dataset
                 row = group_rows[rng.integers(0, len(group_rows))]
                 text = str(row[text_col])

                 aug_texts = augmenter.augment_text(text)
                 if not aug_texts:
                     continue 
                
                 # take one augmentation at a time
                 # by taking the first item of the list
                 aug_text = aug_texts[0]

                 new_row = dict(row)
                 new_row[text_col] = aug_text
                 new_row[label_col] = label 
                
                 # Converts the dict to pdSeries 
                 # Enforces column order
                 # adds this new example to the growing list
                 augmented_rows.append((new_row))

        aug_df = pd.DataFrame(augmented_rows, columns=base_cols)

        # Original + augmented
        output = pd.concat([df, aug_df], axis = 0, ignore_index=True)

        # Shuffle for training stability
        output = output.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
        return output 


                  







