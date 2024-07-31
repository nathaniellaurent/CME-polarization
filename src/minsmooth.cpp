#include <iostream>
#include <valarray>
#include <vector>
#include <string>




using namespace std;


class Solution {



public:
    inline static vector<int> myCosts;
    inline static vector<string> globWords;
    inline static vector<int> globCosts;
    int minimumCost(std::string target, vector<string>& words, vector<int>& costs) {
        globWords = words;
        globCosts = costs;
        nextWord(target, 0);
        return *min_element(myCosts.begin(), myCosts.end());
    }

    void nextWord(string currentTarget, int currentCost){
        for(int i = 0; i < globWords.size(); ++i){
            if(currentTarget.find(globWords[i]) == 0){
                currentTarget = currentTarget.substr(globWords[i].size()+1);
                currentCost += globCosts[i]; 
                if(currentTarget == ""){
                    myCosts.push_back(currentCost);
                }
                else{
                    nextWord(currentTarget, currentCost);
                }

            }
        }
    }




};

int main() {
    Solution sol;
    vector<string> words = {"abc", "ab", "cd", "def", "abcd"};
    vector<int> costs = {1, 2, 3, 4, 5};
    cout << sol.minimumCost("abcdef", words, costs) << endl;
    return 0;
}