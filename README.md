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

## Package Installation
The source code can be compiled in a Python3 IDE. The required packages are:

```sh
numpy
pyslha
```

The easiest way to install these packages is to use the pip install command:

```sh
pip install <package>
```

## Instructions
- Compile the source code in your IDE. You will be prompted to enter the directory containing the output SLHA file from your spectrum calculator. See example below:

![image](https://user-images.githubusercontent.com/85904612/123331182-91584a00-d504-11eb-868f-fdea750dc179.png)

- Use the full directory if your SLHA file is outside of your Python path. See example below:

![image](https://user-images.githubusercontent.com/85904612/123335876-b5b72500-d50a-11eb-8714-43db7c1c7992.png)

- The calculator will then produce the value of DEW, as well as the top ten contributions to DEW. See example below:

![image](https://user-images.githubusercontent.com/85904612/123335984-db442e80-d50a-11eb-8f73-5bb0d0040053.png)

## Descriptions of Source Code:
Below are tables with descriptions of the parameters found in the source code. All mass eigenstate eigenvalues are enumerated in increasing order in magnitude, i.e., |m_1| < |m_2|. Parameters with a description such as 'parameter(Q)' means that the parameter has been evolved to the renormalization scale, Q, with Q taken to be

![image](https://user-images.githubusercontent.com/85904612/123335409-067a4e00-d50a-11eb-88b1-8df6b125055f.png)

| Parameters | Description |
| ------ | ------ |
| vHiggs | Higgs VEV(Q) |
| muQ | Soft SUSY parameter, mu(Q) |
| tanb | Ratio of Higgs VEVs, tan(beta)(Q) |
| y_t | Top Yukawa coupling y_t(Q) |
| y_b | Bottom Yukawa coupling y_b(Q) |
| y_tau | Tau Yukawa coupling y_tau(Q) |
| g_pr | Electroweak gauge coupling constant g'(Q) |
| g_EW | Electroweak gauge coupling constant g(Q) |
| m_stop_1 | Stop mass eigenstate eigenvalue (EV) 1 |
| m_stop_2 | Stop mass eigenstate EV 2 |
| m_sbot_1 | Sbottom mass eigenstate EV 1 |
| m_sbot_2 | Sbottom mass eigenstate EV 2 |
| m_stau_1 | Stau mass eigenstate EV 1 |
| m_stau_2 | Stau mass eigenstate EV 2 |
| mtL | Left stop gauge eigenstate soft scalar mass, mqL3(Q) |
| mtR | Right stop gauge eigenstate soft scalar mass, mtR(Q) |
| mbL | Left sbottom gauge eigenstate soft scalar mass, mqL3(Q) |
| mbR | Right sbottom gauge eigenstate soft scalar mass, mbR(Q) |
| mtauL | Left stau gauge eigenstate soft scalar mass, mtauL(Q) |
| mtauR | Right stau gauge eigenstate soft scalar mass, mtauR(Q) |
| msupL | Left sup soft scalar mass, mqL1(Q) |
| msupR | Right sup soft scalar mass, muR(Q) |
| msdownL | Left sdown soft scalar mass, mqL1(Q) |
| msdownR | Right sdown soft scalar mass, mdR(Q) |
| mselecL | Left selectron soft scalar mass, meL(Q) |
| mselecR | Right selectron soft scalar mass, meR(Q) |
| mselecneut | Selectron sneutrino scalar mass |
| msmuneut | Smuon snuetrino scalar mass |
| msstrangeL | Left sstrange soft scalar mass, mqL2(Q) |
| msstrangeR | Right sstrange soft scalar mass, msR(Q) |
| mscharmL | Left scharm soft scalar mass, mqL2(Q) |
| mscharmR | Right scharm soft scalar mass, mcR(Q) |
| msmuL | Left smuon soft scalar mass, mmuL(Q) |
| msmuR | Right smuon soft scalar mass, mmuR(Q) |
| msN1 | Neutralino mass eigenstate EV 1 |
| msN2 | Neutralino mass eigenstate EV 2 |
| msN3 | Neutralino mass eigenstate EV 3 |
| msN4 | Neutralino mass eigenstate EV 4 |
| msC1 | Chargino mass eigenstate EV 1 |
| msC2 | Chargino mass eigenstate EV 2 |
| mZ | Z boson pole mass |
| mh0 | Lighter neutral Higgs mass |
| mH0 | Heavier neutral Higgs mass |
| mHusq | Up-type Higgs squared mass, mH2^2(Q) |
| mHdsq | Down-type Higgs squared mass, mH1^2(Q) |
| mH_pm | Charged Higgs mass |
| M_1 | Bino mass parameter, M_1(Q) |
| M_2 | Wino mass parameter, M_2(Q) |
| a_t | Trilinear stop scalar coupling, A_t(Q) * y_t(Q) |
| a_b | Trilinear sbottom scalar coupling, A_b(Q) * y_b(Q) |
| a_tau | Trilinear stau scalar coupling, A_tau(Q) * y_tau(Q) |
| Q_renorm | Renormalization scale |
| halfmzsq | Half of pole mass of Z boson squared |
