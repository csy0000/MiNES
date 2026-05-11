# Instruction on 2026-05-02

## [19:35]
This is the first instruction to claude code on this project. The overall project goal is to develop a method "MiNES" to speed up the convergence of the PMF and find the free energy minimal path (FMEP) in a high-dimensional space. The detailed goal is swritten in `legacy/README.md`. And example system configuration is in `legacy/run_context.json`

I already have some files written and will need you to assist me to complete the project. As the first instruction, I would like to set up some principles that you probably want to add to `CLAUDE.md` as the first file to check:
### [Pinciple Rules]
1. Memody: You are allowed to create any thing in `claude-plan/docs` for you to store your memory and understanding the structure of the project.
2. Daily and special and instruction: I will write all my instructions in the folder `claude-plan/yyyy-mm/yyyy-mm-dd-Instruction.md` to tell you what I want to do. Sometimes I might have longer instructions and these will be written in different filename like yyyy-mm-dd-OPERATION.md. In the daily instruction, I will use a time code `##[hh:mm]` to record the time of my instruction
3. Execution: for executing, create files in `claude-plan/yyyy-mm/yyyy-mm-dd-Execution.md` with the time code to specify to which instruction you are answering to.
4. Efficiency: If the instruction does not explicitly ask you to test or execute scripts, write the operation in `claude-plan/yyyy-mm/yyyy-mm-dd-Operation.md` with the time code so I know what I should do to continue. But you are free to create new files in this project.
5. When reading instructions, do not spend time reading instructions which do not belong to the same day except explicitly asked for.
6. Add some more suggestions from `https://github.com/anthropics/claude-code-action/tree/main` if you think that it can help with our project.

### [Understanding the codebase]
Now, all the files in this project and tell me what you understand. There are probably some redundant files, tell me what you would suggest to make the project structure cleaner. At the end of the project I only need
1. 1D examples for 2-well and 3-well potentials
2. 2D examples with Muller-Brown potentials.
For comparing the method, I will compare
(A) Non-adaptive method
1. Umbrella sampling with different parameters: window spacing and k
2. NES sampling with different parameters: switching time, k, and number of switchings
(B) Adaptive method
1. Metadynamics (MTD) with different parameters
2. MiNES, which is now written in `scripts`, with different parameters.

Now execute.

## [20:13]
Now I have moved the entire `analysis` folder here and also put `mines_variance_fusion_visualization.ipynb` in the folder.
1. In this very instruction [20:13], you are permitted to remove anything we don't need to execute the benchmark and merge files if you think that makes sense.
2. We will move on to the 1Dthreewell later. Right now I only want to focus on making MiNUS to run on simple 1Ddoublewell system.

## [20:20]
Also, at the top of each implementation file, note how much tockens have been spent for each time code.

## [20:24]
I have done the following operations: (a) Moved the bash files to scripts/ and (b). move the analysis python files to src/analysis. Now do the following:
1. Adapt the codebase based on this change.
2. Right now, the four comparing methods (US, NES, MTD, MiNES) seem to share some files in the simulation scripts such as `adaptive_methods.py` and the bash scripts in scripts/. I want you to seperate them into different files.
3. Write or update the README files folders analysis/, scripts/, simulations/, and src/


## [20:48]
In the daily operation file, remove the operations that are no longer needed and only leave the ones that are still needed to be done by me.

## [21:42]
The current `analysis/mines_variance_fusion_visualization.ipynb` is often stall. I suspect that some pandas dataframe calling is calling the problem. Please identify the problem and improve it.