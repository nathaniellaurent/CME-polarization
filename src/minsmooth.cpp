#include <iostream>
#include <valarray>
#include <CCfits> 

namespace fit = CCfits;

int main(int argc, char * argv[]) {
    fit::FITS inFile(
        "../Scripts/CME_0_pB/stepnum_005.fits",
        fit::Read,
        true
    );

    fit::PHDU & phdu = inFile.pHDU();

    std::valarray<unsigned int> fitsImage;
    phdu.read(fitsImage);

    return 0;
}