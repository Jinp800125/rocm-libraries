/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <ResultFileReporter.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<ResultFileReporter>
            ResultFileReporter::Default(po::variables_map const& args)
        {
            return std::make_shared<ResultFileReporter>(
                args["results-file"].as<std::string>(),
                args["csv-export-extra-cols"].as<bool>(),
                args["csv-merge-same-problems"].as<bool>(),
                args["performance-metric"].as<PerformanceMetric>());
        }

        ResultFileReporter::ResultFileReporter(std::string const& filename,
                                               bool               exportExtraCols,
                                               bool               mergeSameProblems,
                                               PerformanceMetric  performanceMetric)
            : m_output(filename)
            , m_extraCol(exportExtraCols)
            , m_mergeSameProblems(mergeSameProblems)
            , m_performanceMetric(performanceMetric)
        {
            if(m_performanceMetric == PerformanceMetric::CUEfficiency)
                m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFlopsPerCU");
            else // Default to 'DeviceEfficiency' benchmarking if CUEfficiency not specified
                m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFlops");
        }

        template <typename T>
        void ResultFileReporter::reportValue(std::string const& key, T const& value)
        {
            // if (VICTOR_LOG) std::cout << __PRETTY_FUNCTION__ << std::endl;
            std::string valueStr = boost::lexical_cast<std::string>(value);

            if(key == ResultKey::Validation)
            {
                if(valueStr != "PASSED" && valueStr != "NO_CHECK")
                {
                    m_output.setValueForKey(m_solutionName, -1.0);
                    m_invalidSolution = true;
                }
            }
            else if(key == ResultKey::SolutionName)
            {
                m_solutionName = valueStr;
                m_output.setHeaderForKey(valueStr, valueStr);
            }
            else if(key == ResultKey::SolutionIndex)
            {
                if (VICTOR_LOG)
                    std::cout << __PRETTY_FUNCTION__ << "SolutionIndex" << std::endl;
                m_currSolutionIdx = std::stod(valueStr);
            }
            else if(key == ResultKey::TimeUS)
            {
                if (VICTOR_LOG)
                    std::cout << __PRETTY_FUNCTION__ << "TimeUS" << std::endl;
                // cascade from BenchmarkTimer, Time-US first
                // ++m_currSolutionIdx;
                // m_currSolutionIdx = std::stod(m_output.readValueFromKey(ResultKey::SolutionIndex)); // Victor check
                if(!m_invalidSolution)
                {
                    double timeUS    = std::stod(valueStr);
                    bool   timeIsNan = std::isnan(timeUS);
                    if((!timeIsNan) && (m_fasterTimeUS < 0 || m_fasterTimeUS > timeUS))
                    {
                        m_fasterTimeUS = timeUS;
                        if(m_extraCol)
                        {
                            m_fastestTilesPerCu
                                = std::stod(m_output.readValueFromKey(ResultKey::TilesPerCu));
                            m_fastestTotalGranularity
                                = std::stod(m_output.readValueFromKey(ResultKey::TotalGranularity));
                        }
                    }
                    m_top[timeUS].push_back(m_currSolutionIdx);
                }
            }
            else if((key == ResultKey::SpeedGFlops
                     && m_performanceMetric == PerformanceMetric::DeviceEfficiency)
                    || (key == ResultKey::SpeedGFlopsPerCu
                        && m_performanceMetric == PerformanceMetric::CUEfficiency))
            {
                // cascade from BenchmarkTimer, SpeedGFlops or SpeedGFlopsPerCU second
                if(!m_invalidSolution)
                {
                    m_output.setValueForKey(m_solutionName, value);

                    double gflops = std::stod(valueStr);
                    if(m_fastestGflops < gflops)
                    {
                        m_winnerSolution    = m_solutionName;
                        m_winnerSolutionIdx = m_currSolutionIdx;
                        m_fastestGflops     = gflops;
                    }
                    // m_top[gflops].push_back(m_currSolutionIdx);
                }
            }
            else
            {
                m_output.setValueForKey(key, value);
            }
        }

        void ResultFileReporter::reportValue_string(std::string const& key,
                                                    std::string const& value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_uint(std::string const& key, uint64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_int(std::string const& key, int64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_double(std::string const& key, double value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_sizes(std::string const&         key,
                                                   std::vector<size_t> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                for(size_t i = 0; i < value.size(); i++)
                {
                    std::string key = concatenate("Size", static_cast<char>('I' + i));
                    m_output.setHeaderForKey(key, key);
                    m_output.setValueForKey(key, value[i]);
                }

                // Values for these come separately.
                m_output.setHeaderForKey(ResultKey::LDD, "LDD");
                m_output.setHeaderForKey(ResultKey::LDC, "LDC");
                m_output.setHeaderForKey(ResultKey::LDA, "LDA");
                m_output.setHeaderForKey(ResultKey::LDB, "LDB");
                m_output.setHeaderForKey(ResultKey::TotalFlops, "TotalFlops");
                if(m_extraCol)
                {
                    m_output.setHeaderForKey(ResultKey::TilesPerCu, "TilesPerCu");
                    m_output.setHeaderForKey(ResultKey::TotalGranularity, "TotalGranularity");
                    m_output.setHeaderForKey(ResultKey::FastestGFlops, "WinnerGFlops");
                    m_output.setHeaderForKey(ResultKey::TimeUS, "WinnerTimeUS");
                    m_output.setHeaderForKey(ResultKey::SolutionWinnerIdx, "WinnerIdx");
                    m_output.setHeaderForKey(ResultKey::SolutionWinner, "WinnerName");
                }
            }
        }

        // Remain unchanged
        void ResultFileReporter::reportValue_vecOfSizes(
            std::string const& key, std::vector<std::vector<size_t>> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                for(size_t i = 0; i < value[0].size(); i++)
                {
                    std::string key = concatenate("Size", static_cast<char>('I' + i));
                    m_output.setHeaderForKey(key, key);
                    m_output.setValueForKey(key, value[0][i]);
                }

                // Values for these come separately.
                m_output.setHeaderForKey(ResultKey::LDD, "LDD");
                m_output.setHeaderForKey(ResultKey::LDC, "LDC");
                m_output.setHeaderForKey(ResultKey::LDA, "LDA");
                m_output.setHeaderForKey(ResultKey::LDB, "LDB");
                m_output.setHeaderForKey(ResultKey::TotalFlops, "TotalFlops");

                if(m_extraCol)
                {
                    m_output.setHeaderForKey(ResultKey::TilesPerCu, "TilesPerCu");
                    m_output.setHeaderForKey(ResultKey::TotalGranularity, "TotalGranularity");
                    m_output.setHeaderForKey(ResultKey::FastestGFlops, "WinnerGFlops");
                    m_output.setHeaderForKey(ResultKey::TimeUS, "WinnerTimeUS");
                    m_output.setHeaderForKey(ResultKey::SolutionWinnerIdx, "WinnerIdx");
                    m_output.setHeaderForKey(ResultKey::SolutionWinner, "WinnerName");
                }
            }
        }

        void ResultFileReporter::mergeRow(std::unordered_map<std::string, std::string>& newRow)
        {
            m_currProbID = std::stoull(newRow[ResultKey::ProblemIndex]);
            if(m_probMap.count(m_currProbID) == 0)
            {
                m_probMap[m_currProbID] = newRow;
                return;
            }

            auto& oldRow = m_probMap[m_currProbID];
            for(auto& oldRowIter : oldRow)
            {
                const std::string& key = oldRowIter.first;
                if(key.compare(ResultKey::ProblemIndex) == 0
                   || key.find("Size") != std::string::npos || key.compare(ResultKey::LDD) == 0
                   || key.compare(ResultKey::LDC) == 0 || key.compare(ResultKey::LDA) == 0
                   || key.compare(ResultKey::LDB) == 0 || key.compare(ResultKey::TotalFlops) == 0)
                {
                    // these data should be the same for same problem
                    assert(oldRowIter.second == newRow[key]);
                }
                else if(key.compare(ResultKey::FastestGFlops) == 0)
                {
                    // if new row is better, update, dummy guard for -1 and empty str
                    int64_t oldFastest
                        = (oldRowIter.second.empty()) ? 0 : std::stoll(oldRowIter.second);
                    int64_t newFastest = (newRow[key].empty()) ? 0 : std::stoll(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[ResultKey::FastestGFlops]     = newRow[ResultKey::FastestGFlops];
                        oldRow[ResultKey::TimeUS]            = newRow[ResultKey::TimeUS];
                        oldRow[ResultKey::SolutionWinnerIdx] = newRow[ResultKey::SolutionWinnerIdx];
                        oldRow[ResultKey::SolutionWinner]    = newRow[ResultKey::SolutionWinner];
                        oldRow[ResultKey::TilesPerCu]        = newRow[ResultKey::TilesPerCu];
                        oldRow[ResultKey::TotalGranularity]  = newRow[ResultKey::TotalGranularity];
                    }
                }
                else if(key.compare(ResultKey::TimeUS) == 0
                        || key.compare(ResultKey::SolutionWinnerIdx) == 0
                        || key.compare(ResultKey::SolutionWinner) == 0)
                {
                    // skip, we update these together with FastestGFlops
                    continue;
                }
                else
                {
                    // these are gflops for each solution
                    // if new row is better, update. Dummy guard for -1 and empty str
                    int64_t oldFastest
                        = (oldRowIter.second.empty()) ? 0 : std::stoll(oldRowIter.second);
                    int64_t newFastest = (newRow[key].empty()) ? 0 : std::stoll(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[key] = newRow[key];
                    }
                }
            }
        }

        void ResultFileReporter::postProblem()
        {
            if(m_extraCol)
            {
                // update winner
                m_output.setValueForKey(ResultKey::TilesPerCu, m_fastestTilesPerCu);
                m_output.setValueForKey(ResultKey::TotalGranularity, m_fastestTotalGranularity);
                m_output.setValueForKey(ResultKey::FastestGFlops, m_fastestGflops);
                m_output.setValueForKey(ResultKey::TimeUS, m_fasterTimeUS);
                m_output.setValueForKey(ResultKey::SolutionWinnerIdx, m_winnerSolutionIdx);
                m_output.setValueForKey(ResultKey::SolutionWinner, m_winnerSolution);
            }
            // reset
            m_winnerSolution          = "";
            m_currSolutionIdx         = -1;
            m_winnerSolutionIdx       = -1;
            m_fastestGflops           = -1.0;
            m_fasterTimeUS            = -1.0;
            m_fastestTilesPerCu       = -1.0;
            m_fastestTotalGranularity = -1.0;
            m_top.clear();

            if(!m_mergeSameProblems)
            {
                m_output.writeCurrentRow();
            }
            else
            {
                std::unordered_map<std::string, std::string> curRow;
                m_output.readCurrentRow(curRow);
                m_output.clearCurrentRow();
                // for (auto & field : curRow )
                //     std::cout << "key:" << field.first << ", value:" << field.second << std::endl;
                this->mergeRow(curRow);
            }
        }

        void ResultFileReporter::STEP2resetProblem()
        {
            if (VICTOR_LOG) std::cout << __PRETTY_FUNCTION__ << std::endl;
            // reset
            m_winnerSolution          = "";
            m_currSolutionIdx         = -1;
            m_winnerSolutionIdx       = -1;
            m_fastestGflops           = -1.0;
            m_fasterTimeUS            = -1.0;
            m_fastestTilesPerCu       = -1.0;
            m_fastestTotalGranularity = -1.0;
            m_top.clear();
        }

        void ResultFileReporter::postSolution()
        {
            if (VICTOR_LOG) std::cout << __PRETTY_FUNCTION__ << std::endl;
            m_solutionName    = "";
            m_invalidSolution = false;
        }

        void ResultFileReporter::finalizeReport()
        {
            if(m_mergeSameProblems)
            {
                for(auto& probIter : m_probMap)
                {
                    auto& single_row = probIter.second;
                    for(auto& field : single_row)
                    {
                        m_output.setValueForKey(field.first, field.second);
                    }
                    m_output.writeCurrentRow();
                }
            }
        }

        void ResultFileReporter::setPredictionIdx(int64_t solutionIdx, int64_t predictionIdx)
        {
            // std::cout << __PRETTY_FUNCTION__ << "solutionIdx: " << solutionIdx << " predictionIdx: " << predictionIdx << std::endl;
            m_predictionIdx[solutionIdx] = predictionIdx;
        }

        void ResultFileReporter::getTop(std::vector<int64_t> &v_top, int top_want)
        {
            if (VICTOR_LOG) std::cout << __PRETTY_FUNCTION__ << std::endl;
            int idx;
            auto it = m_top.begin();
            std::cout << "\n";
            // 临时存储 (sloIdx, predictionIdx) 对，用于排序
            std::vector<std::pair<int64_t, int64_t>> temp_pairs;
            int perf_idx = 0;
            for (it = m_top.begin(), idx=0; (it != m_top.end()) && idx<top_want; it++, idx++) {
            // for (const auto& top : m_top) {
                // std::cout << "\n" << "MAP" << top.first << " " << top.second << "\n";
                for (const auto& sloIdx : it->second) {
                    perf_idx++;
                    if (0) std::cout << "\n" << idx << " MAP: " << it->first << " " << sloIdx << "\n";
                    auto predIt = m_predictionIdx.find(sloIdx);
                    int64_t predictionIdx = INT64_MAX; // 默认值，让找不到的排在最后
                    if (predIt != m_predictionIdx.end()) {
                        predictionIdx = predIt->second;
                        if (VICTOR_LOG) std::cout << " " << predIt->second << "(" << perf_idx << ")" << sloIdx;
                    } else {
                        std::cout << " N/A";
                    }
                    // Store perf_idx mapping
                    m_perfIdx[sloIdx] = perf_idx;
                    temp_pairs.push_back({sloIdx, predictionIdx});
                }
            }
            // 根据 predictionIdx 由大到小排序
            std::sort(temp_pairs.begin(), temp_pairs.end(), 
                      [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
                          return a.second > b.second;
                      });
            // 将排序后的 sloIdx 放入 v_top
            v_top.clear();
            for (const auto& pair : temp_pairs) {
                v_top.push_back(pair.first);
            }
            std::cout << "\n";
        }

        int64_t ResultFileReporter::getPerfIdx(int64_t solutionIdx) const
        {
            if (VICTOR_LOG) std::cout << solutionIdx << std::endl;
            auto it = m_perfIdx.find(solutionIdx);
            if (it != m_perfIdx.end()) {
                return it->second;
            }
            return -1;  // Return -1 if not found
        }
    } // namespace Client
} // namespace TensileLite
