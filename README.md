# DEW Calculator
This program computes the naturalness measure Delta_EW (DEW) from an SLHA file provided by the user. The source code can be compiled in a Python3 IDE.

## Calculator Scope
The calculator is designed to be an easy-to-use, quick method for obtaining this naturalness value by operating directly on the output SLHA file from a spectrum calculator (e.g., SoftSUSY, Isajet, SPHENO, etc.).

The calculator will produce the value of Delta_EW, defined in terms of the Higgs one-loop minimization condition:

![image](https://user-images.githubusercontent.com/85904612/123332148-c3b67700-d505-11eb-88c6-1d0aa9cc8488.png)

The naturalness measure Delta_EW is defined as 

![image](https://user-images.githubusercontent.com/85904612/123333435-5d325880-d507-11eb-9437-68f18dbe9657.png)


The contributions, C_i are given as

![image](https://user-images.githubusercontent.com/85904612/123333314-36742200-d507-11eb-9343-4bdaf9272592.png)

Above, k denotes the various individual one-loop contributions to the minimization conditions. 

## Package Installation for Running dew_source_code.py in Python3 IDE
The source code can be compiled in a Python3 IDE. The required packages are:

```sh
numpy
pyslha
```

The easiest way to install these packages is to use the pip install command:

```sh
pip install <package>
```

## Instructions for Running dew_source_code.py in Python3 IDE
- Obtain a SLHA format output file from your choice of spectrum calculator.

- Compile the source code in your IDE. You will be prompted to enter the directory containing the output SLHA file from your spectrum calculator. See example below:

![image](https://user-images.githubusercontent.com/85904612/123331182-91584a00-d504-11eb-868f-fdea750dc179.png)

- Use the full directory if your SLHA file is outside of your Python path. See example below:

![image](https://user-images.githubusercontent.com/85904612/123335876-b5b72500-d50a-11eb-8714-43db7c1c7992.png)

- The calculator will then produce the value of DEW, as well as the top ten contributions to DEW. See example below:

![image](https://user-images.githubusercontent.com/85904612/123335984-db442e80-d50a-11eb-8f73-5bb0d0040053.png)

## Instructions for Running dew_code in Terminal
- Module dependencies are already packaged into dew_calculator.

- Obtain a SLHA format output file from your choice of spectrum calculator.

- Run bash or open up terminal. See example below:

![image](https://user-images.githubusercontent.com/85904612/123345416-9aa0e100-d51b-11eb-980f-5ab3b5545537.png)

- cd to location of dew_code file. See example below:

![image](https://user-images.githubusercontent.com/85904612/123344575-c4590880-d519-11eb-8e0b-8ba1ab1abf2b.png)

- Run the following command: 
```sh
./dew_calculator
```

- Input the directory of your SLHA file when prompted. See example below:

![image](https://user-images.githubusercontent.com/85904612/123346470-03895880-d51e-11eb-9168-686a081c2513.png)

- The calculator will then produce the value of DEW, as well as the top ten contributions to DEW. See example below:

![image](https://user-images.githubusercontent.com/85904612/123346509-1865ec00-d51e-11eb-84ca-3e332b8826d2.png)

## Descriptions of Source Code:
Below are tables with descriptions of the parameters found in the source code. All mass eigenstate eigenvalues are enumerated in increasing order in magnitude, i.e., |m_1| < |m_2|. Parameters with a description such as 'parameter(Q)' means that the parameter has been evolved to the renormalization scale, Q, with Q taken to be:

![image](https://user-images.githubusercontent.com/85904612/123335409-067a4e00-d50a-11eb-88b1-8df6b125055f.png)

SLHA Block information is presented as ['BLOCKNAME']['ENTRY ID #'].

| Parameters | Description | SLHA Block information |
| ------ | ------ | ------ |
| vHiggs | Higgs VEV(Q) | ['HMIX'][3] |
| muQ | Soft SUSY parameter, mu(Q) | ['HMIX'][1] |
| tanb | Ratio of Higgs VEVs, tan(beta)(Q) | ['HMIX'][2] |
| y_t | Top Yukawa coupling y_t(Q) | ['YU'][3, 3] |
| y_b | Bottom Yukawa coupling y_b(Q) | ['YD'][3, 3] |
| y_tau | Tau Yukawa coupling y_tau(Q) | ['YE'][3, 3] |
| g_pr | Electroweak gauge coupling constant g'(Q) | ['GAUGE'][2] |
| g_EW | Electroweak gauge coupling constant g(Q) | ['GAUGE'][1] |
| m_stop_1 | Stop mass eigenstate eigenvalue (EV) 1 | ['MASS'][1000006] |
| m_stop_2 | Stop mass eigenstate EV 2 | ['MASS'][2000006] |
| m_sbot_1 | Sbottom mass eigenstate EV 1 | ['MASS'][1000005] |
| m_sbot_2 | Sbottom mass eigenstate EV 2 | ['MASS'][2000005] |
| m_stau_1 | Stau mass eigenstate EV 1 | ['MASS'][1000015] |
| m_stau_2 | Stau mass eigenstate EV 2 | ['MASS'][2000015] |
| mtL | Left stop gauge eigenstate soft scalar mass, mqL3(Q) | ['MSOFT'][43] |
| mtR | Right stop gauge eigenstate soft scalar mass, mtR(Q) | ['MSOFT'][46] |
| mbL | Left sbottom gauge eigenstate soft scalar mass, mqL3(Q) | ['MSOFT'][43] |
| mbR | Right sbottom gauge eigenstate soft scalar mass, mbR(Q) | ['MSOFT'][49] |
| mtauL | Left stau gauge eigenstate soft scalar mass, mtauL(Q) | ['MSOFT'][33] |
| mtauR | Right stau gauge eigenstate soft scalar mass, mtauR(Q) | ['MSOFT'][36] |
| msupL | Left sup soft scalar mass, mqL1(Q) | ['MSOFT'][41] |
| msupR | Right sup soft scalar mass, muR(Q) | ['MSOFT'][44] |
| msdownL | Left sdown soft scalar mass, mqL1(Q) | ['MSOFT'][41] |
| msdownR | Right sdown soft scalar mass, mdR(Q) | ['MSOFT'][47] |
| mselecL | Left selectron soft scalar mass, meL(Q) | ['MSOFT'][31] |
| mselecR | Right selectron soft scalar mass, meR(Q) | ['MSOFT'][34] |
| mselecneut | Selectron sneutrino scalar mass | ['MASS'][1000012] |
| msmuneut | Smuon snuetrino scalar mass | ['MASS'][1000014] |
| msstrangeL | Left sstrange soft scalar mass, mqL2(Q) | ['MSOFT'][42] |
| msstrangeR | Right sstrange soft scalar mass, msR(Q) | ['MSOFT'][48] |
| mscharmL | Left scharm soft scalar mass, mqL2(Q) | ['MSOFT'][42] |
| mscharmR | Right scharm soft scalar mass, mcR(Q) | ['MSOFT'][45] |
| msmuL | Left smuon soft scalar mass, mmuL(Q) | ['MSOFT'][32] |
| msmuR | Right smuon soft scalar mass, mmuR(Q) | ['MSOFT'][35] |
| msN1 | Neutralino mass eigenstate EV 1 | ['MASS'][1000022] |
| msN2 | Neutralino mass eigenstate EV 2 | ['MASS'][1000023] |
| msN3 | Neutralino mass eigenstate EV 3 | ['MASS'][1000025] |
| msN4 | Neutralino mass eigenstate EV 4 | ['MASS'][1000035] |
| msC1 | Chargino mass eigenstate EV 1 | ['MASS'][1000024] |
| msC2 | Chargino mass eigenstate EV 2 | ['MASS'][1000037] |
| mZ | Z boson pole mass | ['SMINPUTS'][4] |
| mh0 | Lighter neutral Higgs mass | ['MASS'][25] |
| mH0 | Heavier neutral Higgs mass | ['MASS'][35] |
| mHusq | Up-type Higgs squared mass, mH2^2(Q) | ['MSOFT'][22] |
| mHdsq | Down-type Higgs squared mass, mH1^2(Q) | ['MSOFT'][21] |
| mH_pm | Charged Higgs mass | ['MASS'][37] |
| M_1 | Bino mass parameter, M_1(Q) | ['MSOFT'][1] |
| M_2 | Wino mass parameter, M_2(Q) | ['MSOFT'][2] |
| a_t | Trilinear stop scalar coupling, A_t(Q) * y_t(Q) | A_t(Q): ['AU'][3, 3] |
| a_b | Trilinear sbottom scalar coupling, A_b(Q) * y_b(Q) | A_b(Q): ['AD'][3, 3] |
| a_tau | Trilinear stau scalar coupling, A_tau(Q) * y_tau(Q) | A_tau(Q): ['AE'][3, 3] |
| Q_renorm | Renormalization scale | N/A |
| halfmzsq | Half of pole mass of Z boson squared | N/A |
