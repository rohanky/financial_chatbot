from rank_bm25 import BM25Okapi
import pandas as pd
import faiss
import numpy as np
import os
import re
from openai import AzureOpenAI

EMBEDDING_MODEL = "text-embedding-3-large"  # Adjust if necessary

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Define stop words for keyword-based search
STOP_WORDS = set([
"0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",
])


class Retriever:
    def __init__(self, index_file=None, data_file=None):
        """
        Initialize the retriever by loading data and FAISS index.
        """
        self.embedding_dimension = 1024  # Adjust based on your embedding model
        self.data = self.load_and_group_data(data_file)
        self.bm25_model = self.initialize_bm25()
        self.index = self.load_or_create_index(index_file)

    def load_and_group_data(self, file_path, split_length=1000):
        """
        Load and group data by Company, File Name, and Page Number for PDFs,
        and by Company and File Name for HTML files.
        Split content into smaller chunks of specified length.
        """
        df = pd.read_excel(file_path)

        # Check if 'File Type' column exists
        if 'File Type' not in df.columns:
            raise ValueError("The input file must have a 'File Type' column indicating 'pdf' or 'html'.")

        # Separate data for PDFs and HTMLs
        pdf_df = df[df['File Type'] == 'pdf']
        html_df = df[df['File Type'] == 'html']

        # Group PDF data by Company, File Name, and Page Number
        grouped_pdf_df = (
            pdf_df.groupby(['Company', 'File Name', 'Page Number'], as_index=False)
            .agg({'Content': lambda x: ' '.join(x.dropna().astype(str))})
        )

        # Group HTML data by Company and File Name only
        grouped_html_df = (
            html_df.groupby(['Company', 'File Name'], as_index=False)
            .agg({'Content': lambda x: ' '.join(x.dropna().astype(str))})
        )

        # Add a "Page Number" column to HTML data with a placeholder (e.g., "N/A")
        grouped_html_df['Page Number'] = "N/A"

        # Combine grouped PDF and HTML data
        combined_df = pd.concat([grouped_pdf_df, grouped_html_df], ignore_index=True)

        # Split content into smaller chunks
        split_rows = []
        for _, row in combined_df.iterrows():
            content = row['Content']
            if len(content) > split_length:
                parts = [content[i:i + split_length] for i in range(0, len(content), split_length)]
                for idx, part in enumerate(parts):
                    split_rows.append({
                        'Company': row['Company'],
                        'File Name': row['File Name'],
                        'Page Number': f"{row['Page Number']} - Part {idx + 1}",
                        'Content': part
                    })
            else:
                split_rows.append(row)

        # Convert split rows back to DataFrame
        split_df = pd.DataFrame(split_rows)

        return split_df

    def initialize_bm25(self):
        """
        Initialize the BM25 model with the text data.
        """
        tokenized_corpus = self.data['Content'].apply(self.tokenize).tolist()
        return BM25Okapi(tokenized_corpus)

    def tokenize(self, text):
        """
        Tokenize text for BM25, removing stop words.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in STOP_WORDS]

    def generate_embeddings(self, texts):
        """
        Generate text embeddings using Azure OpenAI Embedding API.
        """
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = np.array([embedding.embedding for embedding in response.data])
        return embeddings

    def generate_embedding_for_row(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return np.zeros((self.embedding_dimension,))
        return self.generate_embeddings([text])[0]

    def build_faiss_index(self):
        """
        Build FAISS index from grouped data.
        """
        self.data['full_content'] = "This article is about: " + self.data['File Name'] + "\n Content: " + self.data['Content']
        self.data['Embeddings'] = self.data['full_content'].apply(self.generate_embedding_for_row)

        embeddings = np.vstack(self.data['Embeddings'].tolist())
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def load_or_create_index(self, index_file):
        if index_file and os.path.exists(index_file):
            return faiss.read_index(index_file)
        else:
            return self.build_faiss_index()

    def save_faiss_index(self, index_file):
        directory = os.path.dirname(index_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        faiss.write_index(self.index, index_file)

    def faiss_search(self, query, top_k=10):
        """
        Perform semantic search using FAISS.
        """
        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({
                "Company": self.data.iloc[idx]['Company'],
                "File Name": self.data.iloc[idx]['File Name'],
                "Page Number": self.data.iloc[idx]['Page Number'],
                "Content": self.data.iloc[idx]['Content'],
                "FAISS Score": 1 - dist  # Convert distance to similarity
            })
        return results

    def bm25_search(self, query, top_k=10):
        """
        Perform BM25-based search.
        """
        query_tokens = self.tokenize(query)
        scores = self.bm25_model.get_scores(query_tokens)

        # Get top_k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "Company": self.data.iloc[idx]['Company'],
                "File Name": self.data.iloc[idx]['File Name'],
                "Page Number": self.data.iloc[idx]['Page Number'],
                "Content": self.data.iloc[idx]['Content'],
                "BM25 Score": scores[idx]
            })
        return results

    def hybrid_search(self, query, top_k=10, faiss_weight=0.5):
        """
        Perform a hybrid search combining FAISS and BM25 methods.
        """
        faiss_results = self.faiss_search(query, top_k * 2)  # Retrieve more to account for overlap
        bm25_results = self.bm25_search(query, top_k * 2)

        # Create a combined score for both results
        all_results = []

        # Add FAISS results
        for result in faiss_results:
            result['Combined Score'] = faiss_weight * result['FAISS Score']
            all_results.append(result)

        # Add BM25 results
        for bm25_result in bm25_results:
            # Try to match BM25 result with FAISS result
            matched_result = next(
                (r for r in all_results if 
                 r['File Name'] == bm25_result['File Name'] and 
                 r['Page Number'] == bm25_result['Page Number']),
                None
            )
            if matched_result:
                matched_result['Combined Score'] += (1 - faiss_weight) * bm25_result['BM25 Score']
            else:
                bm25_result['Combined Score'] = (1 - faiss_weight) * bm25_result['BM25 Score']
                all_results.append(bm25_result)

        # Sort by combined score
        all_results = sorted(all_results, key=lambda x: x['Combined Score'], reverse=True)
        return all_results[:top_k]
