from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Math knowledge base — formulas, templates, common mistakes
docs = [
    Document(page_content="Quadratic formula: x = (-b ± √(b²-4ac)) / 2a. Used to solve ax²+bx+c=0. Discriminant D=b²-4ac: D>0 two real roots, D=0 one root, D<0 no real roots.", metadata={"topic": "algebra"}),
    Document(page_content="Arithmetic Progression (AP): nth term = a + (n-1)d. Sum of n terms = n/2 × (2a + (n-1)d). Where a=first term, d=common difference.", metadata={"topic": "algebra"}),
    Document(page_content="Geometric Progression (GP): nth term = a×r^(n-1). Sum of n terms = a(1-rⁿ)/(1-r). Sum of infinite GP = a/(1-r) when |r|<1.", metadata={"topic": "algebra"}),
    Document(page_content="Binomial theorem: (a+b)ⁿ = Σ C(n,r) × a^(n-r) × bʳ. General term T(r+1) = C(n,r) × a^(n-r) × bʳ.", metadata={"topic": "algebra"}),
    Document(page_content="Derivative rules: d/dx(xⁿ) = nxⁿ⁻¹. d/dx(sin x) = cos x. d/dx(cos x) = -sin x. d/dx(eˣ) = eˣ. d/dx(ln x) = 1/x. Chain rule: d/dx[f(g(x))] = f'(g(x))×g'(x).", metadata={"topic": "calculus"}),
    Document(page_content="Limits: lim(x→0) sin(x)/x = 1. lim(x→0) (1+x)^(1/x) = e. L'Hopital's rule: if 0/0 or ∞/∞ form, differentiate numerator and denominator separately.", metadata={"topic": "calculus"}),
    Document(page_content="Integration: ∫xⁿdx = xⁿ⁺¹/(n+1) + C. ∫sin(x)dx = -cos(x) + C. ∫cos(x)dx = sin(x) + C. ∫eˣdx = eˣ + C. ∫(1/x)dx = ln|x| + C.", metadata={"topic": "calculus"}),
    Document(page_content="Optimization: At maxima/minima f'(x)=0. If f''(x)<0 it's a maximum. If f''(x)>0 it's a minimum.", metadata={"topic": "calculus"}),
    Document(page_content="Probability basics: P(A∪B) = P(A)+P(B)-P(A∩B). P(A|B) = P(A∩B)/P(B). Independent events: P(A∩B) = P(A)×P(B).", metadata={"topic": "probability"}),
    Document(page_content="Bayes theorem: P(A|B) = P(B|A)×P(A) / P(B). Used to update probability based on new evidence.", metadata={"topic": "probability"}),
    Document(page_content="Binomial distribution: P(X=r) = C(n,r) × pʳ × (1-p)^(n-r). Mean = np. Variance = np(1-p).", metadata={"topic": "probability"}),
    Document(page_content="Expected value E(X) = Σ x×P(X=x). Variance Var(X) = E(X²) - [E(X)]². Standard deviation = √Variance.", metadata={"topic": "probability"}),
]

print("Uploading to Pinecone...")
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME"),
)
print("Done! Knowledge base ready.")