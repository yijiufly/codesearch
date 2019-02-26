#include <map>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void fillHashMap(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    const int n,
    const size_t wordCount,
    // Slices are at most 64 bits
    std::map<uint64_t, std::vector<uint64_t> >& sliceMap,
    const uint64_t firstBit,
    const uint64_t sliceLength,
    const uint64_t slice);

void logger(std::string out) {
    FILE *logFile = fopen("/rhome/lgao027/bigdata/emb/knn/logfile.txt","a");
    fprintf(logFile,"%s\n", out.c_str());
    fclose(logFile);
}

// Struct for Sorting cells by similarity
class SimilarCell {
public:
    uint64_t mismatchingBits;
    uint64_t cell;

    // Constructors.
    SimilarCell() : mismatchingBits(0), cell(0) {}
    SimilarCell(uint64_t mismatchingBits, uint64_t cell) : mismatchingBits(mismatchingBits), cell(cell) {}

    // Comparisons for unique
    bool operator==(const SimilarCell& other){ return this->cell == other.cell; }
    bool operator!=(const SimilarCell& other){ return !(*this == other); }

};

// Compare function to  sort SimilarCells
bool compareSimilarCells(SimilarCell i, SimilarCell j) { return (i.mismatchingBits < j.mismatchingBits); }

// Class that stores pointers to the begin and end of a bit set.
// Does not own the memory, and copies are shallow copies.
// Implements low level operations on bit sets.
class BitSetPointer {
public:

    // Begin and and pointers of the bit set.
    const uint64_t* begin;
    const uint64_t* end;

    // Constructors.
    BitSetPointer(const uint64_t* begin=0, const uint64_t* end=0) : begin(begin), end(end) {}
    BitSetPointer(const uint64_t* begin, uint64_t wordCount) : begin(begin), end(begin+wordCount) {}

    // Return the number of 64-bit words in the bit set.
    uint64_t wordCount() const { return end - begin; }

};

// Count the number of mismatching bits between two bit vectors.
// Use SSE 4.2 builtin instruction popcount
inline uint64_t countMismatches(
    const BitSetPointer& x,
    const BitSetPointer& y)
{
    const uint64_t wordCount = x.wordCount();
    uint64_t mismatchCount = 0;
    for(uint64_t i = 0; i < wordCount; i++) {
        mismatchCount += __builtin_popcountll(x.begin[i] ^ y.begin[i]);
    }
    return mismatchCount;
}

// Compute the similarity (cosine of the angle) corresponding to each number of mismatching bits.
// NOTE: actually we could compute only up to the threshold, the rest we discard anyway
uint64_t computeSimilarityTable(
    const size_t lshCount,
    std::vector<double>& similarityTable,
    double threshold)
{
    // Initialize the similarity table.
    similarityTable.resize(lshCount + 1);

    uint64_t mismatchingBitsThreshold = lshCount;
    bool thresholdFound = false;

    // Loop over all possible numbers of mismatching bits.
    for(size_t mismatchingBitCount = 0;
        mismatchingBitCount <= lshCount;
        mismatchingBitCount++) {

        // Compute the angle between the vectors corresponding to
        // this number of mismatching bits.
        const double angle = double(mismatchingBitCount) *
            3.14159265359 / double(lshCount);

        // The cosine of the angle is the similarity for
        // this number of mismatcning bits.
        const double cosAngle = std::cos(angle);
        similarityTable[mismatchingBitCount] = cosAngle;

        if ((!thresholdFound) && (cosAngle <= threshold)) {
            mismatchingBitsThreshold = mismatchingBitCount;
            thresholdFound = true;
        }
    }

    return mismatchingBitsThreshold;

}

// Fill knn output matrix
void fillOutputMatrices(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    std::vector<double>& similarityTable,
    uint64_t k,
    std::vector<SimilarCell>& candidates,
    uint64_t cellFocal) {

    // Fill output matrix
    uint64_t knni = 0;
    for(std::vector<SimilarCell>::iterator knnit=candidates.begin();
        (knnit != candidates.end()) && (knni < (uint64_t)k);
        knnit++) {
        knn(cellFocal, knni) = knnit->cell;
        similarity(cellFocal, knni) = similarityTable[knnit->mismatchingBits];
        knni++;
    }
    nNeighbors(0, cellFocal) = candidates.size() < (uint64_t)k ? candidates.size() : k;

}

// Fill knn output matrix
void storeOutputToDisk(
    std::string knn,
    std::vector<double>& similarityTable,
    uint64_t k,
    std::vector<SimilarCell>& candidates,
    uint64_t cellFocal) {

    FILE *knnFile = fopen(knn.c_str(), "a");
    // Fill output matrix
    uint64_t knni = 0;
    for(std::vector<SimilarCell>::iterator knnit=candidates.begin();
        (knnit != candidates.end()) && (knni < (uint64_t)k);
        knnit++) {
        fprintf(knnFile, "%"PRIu64"\t%"PRIu64"\t%lf\n", cellFocal, knnit->cell, similarityTable[knnit->mismatchingBits]);
        knni++;
    }
    fclose(knnFile);
}
// Compute k nearest neighbors and similarity values naively, i.e.
// iterating over all pairs
void computeNeighborsViaAllPairs(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    //py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int k,
    const size_t wordCount,
    const int index,
    std::vector<double>& similarityTable,
    const uint64_t mismatchingBitsThreshold) {

    // Hash map
    //std::map<uint64_t, std::vector<uint64_t> > sliceMap;
    //fillHashMap(signature, n, wordCount, sliceMap, 0, 64, index);
    std::vector<SimilarCell> candidates;
    logger(std::to_string(mismatchingBitsThreshold));
    // Calculate all similarities with this cell
    for (uint64_t cell1=0; cell1 < (uint64_t)n; cell1++) {
        candidates.clear();
        BitSetPointer bp1(signature.data() + cell1 * wordCount, wordCount);
        for (uint64_t cell2=0; cell2 < (uint64_t)n; cell2++) {
            if (cell2 == cell1)
                continue;
            BitSetPointer bp2(signature.data() + cell2 * wordCount, wordCount);
            uint64_t nMismatchingBits = countMismatches(bp1, bp2);
            if (nMismatchingBits <= mismatchingBitsThreshold) {
                candidates.push_back({ nMismatchingBits, cell2 });
	    }
        }

	
        // Sort cells by similarities
        if (candidates.size() > (uint64_t)k) {
            std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(), compareSimilarCells);
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(), compareSimilarCells);
	
        /*
        std::cout << "cell1 neighbors sorted for cell " << cell1 << ":\n";
            for (std::vector<SimilarCell>::iterator it=candidates.begin();
             it != candidates.end(); it++) {
            std::cout << (it->cell);
        }
        std::cout << "\n";
        */

        // Fill output matrix
        //fillOutputMatrices(knn, similarity, nNeighbors, similarityTable, k, candidates, cell1);
	
        //char *knnFilePath = "/rhome/lgao027/bigdata/emb/knn_output/candidates.txt";
        std::string knnFilePath = "/rhome/lgao027/bigdata/emb/knn_final/";
	knnFilePath.append(std::to_string(index)).append("/knn/candidates.txt");
        storeOutputToDisk(knnFilePath, similarityTable, k, candidates, cell1);
    }

}

void computeNeighborsForQuery(py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    const int n,
    const int k,
    const size_t wordCount,
    const int index,
    std::vector<double>& similarityTable,
    const uint64_t mismatchingBitsThreshold,
    const int query) {

    std::vector<SimilarCell> candidates;
    logger(std::to_string(mismatchingBitsThreshold));
    // Calculate all similarities with this cell
    for (uint64_t cell1=n-query; cell1 < (uint64_t)n; cell1++) {
        candidates.clear();
        BitSetPointer bp1(signature.data() + cell1 * wordCount, wordCount);
        for (uint64_t cell2=0; cell2 < (uint64_t)n; cell2++) {
            if (cell2 == cell1)
                continue;
            BitSetPointer bp2(signature.data() + cell2 * wordCount, wordCount);
            uint64_t nMismatchingBits = countMismatches(bp1, bp2);
            if (nMismatchingBits <= mismatchingBitsThreshold) {
                candidates.push_back({ nMismatchingBits, cell2 });
	    }
        }

	
        // Sort cells by similarities
        if (candidates.size() > (uint64_t)k) {
            std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(), compareSimilarCells);
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(), compareSimilarCells);

        std::string knnFilePath = "/rhome/lgao027/bigdata/emb/knn_final/";
	knnFilePath.append(std::to_string(index)).append("/knn/query_candidates.txt");
        storeOutputToDisk(knnFilePath, similarityTable, k, candidates, cell1);
    }

}

// Slice multibit signatures for LSH computation
// It is much easier if slices are 64 bit long or less; this is a fair
// requirement for our purposes as having more than 2**64 buckets is useless
uint64_t sliceSignature(uint64_t *data, uint64_t firstBit, uint64_t nBits) {
    if (nBits > 64) {
        throw std::runtime_error("Slices must be at most 64 bit long");
    } else if (nBits == 0) {
        throw std::runtime_error("Slices must be at least 1 bit long");
    }

    // Find whether we are crossing word boundary
    uint64_t firstWord = firstBit >> 6;
    uint64_t lastWord = (firstBit + nBits) >> 6;

    // Not crossing word boundary is easy, just shift and bitwise & and shift back
    if (lastWord == firstWord) {
        // e.g. if nBits = 3 and firstBit = 66, we go to the second word,
        // then shift = 2, so other is b11100 = 28
        // after the bitwise &, the first shift bits are all 0 anyway, we can
        // trash them to be consistent
        uint64_t shift = firstBit % 64;
        uint64_t other = ((1ULL << nBits) - 1ULL) << shift;
        return ((*(data + firstWord)) & other) >> shift;

    // Else, we have to split the job in two
    } else {
        uint64_t shift = firstBit % 64;
        uint64_t nBitsFirst = 64 - shift;
        uint64_t other = ((1ULL << nBitsFirst) - 1ULL) << shift;
        //uint64_t out = ((*(data + firstWord)) & other) >> shift;
        //when shift=0, nBitsFirst=64, it will overflow, so 
	uint64_t out = (*(data + firstWord)) >> shift;

	// The second word has no shift
        uint64_t nBitsSecond = nBits - nBitsFirst;
        other = ((1ULL << nBitsSecond) - 1ULL) << 0;
        out |= ((*(data + lastWord)) & other) << nBitsFirst;
        return out;
    }

}

// Fill hash map of slice
void fillHashMap(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    const int n,
    const size_t wordCount,
    // Slices are at most 64 bits
    std::map<uint64_t, std::vector<uint64_t> >& sliceMap,
    const uint64_t firstBit,
    const uint64_t sliceLength,
    const uint64_t slice) {

    // Clear hash map
    sliceMap.clear();
    logger("fillHashMap");
    // Iterate over cells
    for (uint64_t cell=0; cell < (uint64_t)n; cell++) {
        uint64_t *cellData = (uint64_t*)(signature.data()) + cell * wordCount;
        uint64_t cellHash = sliceSignature(cellData, firstBit, sliceLength);
        //logger(std::to_string(cellHash));
	sliceMap[cellHash].push_back(cell);
	std::string str = "/rhome/lgao027/bigdata/emb/knn_final/";
	str.append(std::to_string(slice)).append("/buckets/").append(std::to_string(cellHash)).append(".txt");
	//FILE *logFile = fopen("/rhome/lgao027/bigdata/emb/knn/"+ std::to_string(slice).c_str() + "/" + std::to_string(cellHash).c_str() +".txt","a");
	FILE *bucket = fopen(str.c_str(), "a");
	fprintf(bucket,"%"PRIu64"\n", cell);
	fclose(bucket);
    }
}


void computeNeighborsViaSlices(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    //py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int k,
    const size_t wordCount,
    std::vector<double>& similarityTable,
    const uint64_t mismatchingBitsThreshold,
    const uint64_t sliceLength,
    const size_t m) {

    // Candidate neighbors
    //std::vector< std::vector<SimilarCell> > neighbors(n);

    // Hash map
    std::map<uint64_t, std::vector<uint64_t> > sliceMap;

    // Iterate over slices
    uint64_t nSlices = m / sliceLength;
    for (uint64_t slice=0; slice < nSlices; slice++) {
        logger("slice");
	// Make hash map for all cells
        fillHashMap(signature, n, wordCount, sliceMap, slice * sliceLength, sliceLength, slice);
    

        // Iterate over map and calculate similarities within each bucket
        for(std::map<uint64_t, std::vector<uint64_t> >::iterator mit=sliceMap.begin();
            mit != sliceMap.end(); mit++) {

            // Naive algorithm, recalculate distances from each cell
            std::vector<uint64_t> cellsBucket = mit->second;
            for(std::vector<uint64_t>::iterator cit=cellsBucket.begin();
                cit != cellsBucket.end();
                cit++) {

                BitSetPointer bp1(signature.data() + (*cit) * wordCount, wordCount);
                std::vector<SimilarCell> candidates;

                for(std::vector<uint64_t>::iterator cit2 = cellsBucket.begin();
                    cit2 != cellsBucket.end();
                    cit2++) {

                    // Skip self
                    if (cit2 == cit)
                        continue;

                    // Skip cell if it is already in the candidates list
                    bool skip = false;
                    for(std::vector<SimilarCell>::iterator candit=candidates.begin();
                        candit != candidates.end();
                        candit++) {
                        if (candit->cell == (*cit2)) {
                            skip = true;
                            break;
                        }
                    }
                    if (skip)
                        break;

                    // Else, calculate similarity
                    BitSetPointer bp2(signature.data() + (*cit2) * wordCount, wordCount);
                    uint64_t nMismatchingBits = countMismatches(bp1, bp2);
                    if (nMismatchingBits <= mismatchingBitsThreshold){
                        candidates.push_back({ nMismatchingBits, *cit2 });
			/*
			std::string str = "/rhome/lgao027/bigdata/emb/knn/";
			str.append(std::to_string(slice)).append("/").append(std::to_string(*cit)).append(".txt");
			FILE *bucket = fopen(str.c_str(), "a");
			fprintf(bucket,"%"PRIu64"\t%"PRIu64"\n", *cit2, nMismatchingBits);
			fclose(bucket);

			std::string str2 = "/rhome/lgao027/bigdata/emb/knn/";
			str2.append(std::to_string(slice)).append("/").append(std::to_string(*cit2)).append(".txt");
			FILE *bucket2 = fopen(str2.c_str(), "a");
			fprintf(bucket2,"%"PRIu64"\t%"PRIu64"\n", *cit, nMismatchingBits);
			fclose(bucket2);*/
		    }
                }

                // Sort candidates and take only top k
                if (candidates.size() > (uint64_t)k) {
                    std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(), compareSimilarCells);
                    candidates.resize(k);
                }
		std::string str = "/rhome/lgao027/bigdata/emb/knn/";
		str.append(std::to_string(slice)).append("/candidates.txt");
		storeOutputToDisk(str, similarityTable, k, candidates, *cit);
            	candidates.clear();
	    }
        }
    }

    // Prepare and write output
    /*uint64_t cellFocal = 0;
    for(std::vector< std::vector<SimilarCell> >::iterator nit=neighbors.begin();
        nit != neighbors.end();
        nit++) {

        // final sort for candidates (they are already <= k)
        std::vector<SimilarCell>* candidates = &(*nit);
        std::sort(candidates->begin(), candidates->end(), compareSimilarCells);

        // Fill output matrix
        //fillOutputMatrices(knn, similarity, nNeighbors, similarityTable, k, *candidates, cellFocal);
        char *knnFilePath = "/rhome/lgao027/bigdata/emb/knn/candidates.txt";
        storeOutputToDisk(knnFilePath, similarityTable, k, *candidates, cellFocal);

        cellFocal++;
    }*/
}


///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
void knn_from_signature(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    //py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    //py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int m,
    const int k,
    const double threshold,
    const int sliceLength,
    const int index,
    const int query) {

    logger("knn begin");
    // signature is a vector containing 64 bit integers for all n cells, the number of 64bit for each cell is
    size_t wordCount = 1 + ((m - 1) / 64);
    // so we can parse 0 to (wordCount - 1) for cell 1, wordCount to (2 * wordCount - 1) for cell 2, etc.

    // Compute the similarity table with m bits
    std::vector<double> similarityTable;
    uint64_t mismatchingBitsThreshold = computeSimilarityTable((size_t)m, similarityTable, threshold);

    logger(std::to_string(mismatchingBitsThreshold));
    // Slower version, go through n^2 pairs
    if (sliceLength == 0) {
    	if (query == 0)
        	computeNeighborsViaAllPairs(
        	signature, //knn, similarity, nNeighbors,
        	n, k, wordCount, index,
        	similarityTable,
        	mismatchingBitsThreshold);
	else{
		computeNeighborsForQuery(
		signature,
		n, k, wordCount, index,
		similarityTable,
		mismatchingBitsThreshold,
		query);
	}

    // Faster version
    } else {
        // 1. Make non-overlapping q bit slices, total m / q
        //    Up to 2^q hashes per group
        // 2. For each subgroup, hash cells
        // 3.     For each cell, find k neighbors in the same hash group
        // 4. Sort cell neighbours from all subgroups and take first k
        // 5. Format for returning
        computeNeighborsViaSlices(
            signature, //knn, similarity, nNeighbors,
            n, k, wordCount,
            similarityTable,
            mismatchingBitsThreshold,
            sliceLength,
            (size_t)m);
    }
}

PYBIND11_MODULE(_lshknn, m) {
    m.def("knn_from_signature", &knn_from_signature, R"pbdoc(
        Add to an existing matrix.

        This tests whether we can access numpy array from Eigen.
        Some other explanation about the add function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
