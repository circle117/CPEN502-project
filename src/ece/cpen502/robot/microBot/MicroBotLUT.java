package ece.cpen502.robot.microBot;


import ece.cpen502.LookUpTable;
import ece.cpen502.robot.Action;
import ece.cpen502.robot.State;
import robocode.*;

import java.io.IOException;

public class MicroBotLUT extends MicroBot {

    private double epsilon = 0.9;
    private final static double epsilonInitial = 0.9;
    private final int totalRoundNum = 2000;
    private final double alpha = 0.5;
    private final double gamma = 0.99;
    private final boolean ifTrain = true;


    static private LookUpTable lookUpTable;
    public MicroBotLUT() {
        super();
        lookUpTable = new LookUpTable(State.EnergyLabel.values().length,
                State.DistanceLabel.values().length,
                State.EnergyLabel.values().length,
                State.DistanceLabel.values().length,
                Action.ActionLabel.values().length,
                alpha, gamma);
    }

    @Override
    public void run() {
        recordFileName = getDataDirectory() +
                String.format("/data-e%.1f-round%d-alpha%.1f-gamma%.2f.csv", epsilon, totalRoundNum, alpha, gamma);
        modelFileName = getDataDirectory() +
                String.format("/lut-e%.1f-round%d-alpha%.1f-gamma%.2f.csv", epsilonInitial, totalRoundNum, alpha, gamma);
        super.run();
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        double turn = getHeading() - getGunHeading() + event.getBearing();
        setTurnGunRight(normalizeBearing(turn));
        if (enemy.none() | event.getName().equals(enemy.getName())) {
            enemy.update(event);
        }

        double[] index = state.indexForLUT(this, event);
        if (lookUpTable.getPreAction()!=-1) rewards.add(reward);

        if (ifTrain) {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.train(index, new double[]{reward})[0]));
                action.doAction(this, event);
                reward = 0;
            } catch (NullPointerException exception) {
                out.println(exception);
            }
        } else {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.outputFor(index)[0]));
                action.doAction(this, event);
            } catch (NullPointerException exception) {
                out.println(exception);
            }
        }
    }


    @Override
    protected void terminal(double terminalReward) throws IOException {
        lookUpTable.updateTerminal(terminalReward);
        super.terminal(terminalReward);
        lookUpTable.setEpsilon(epsilon);
        lookUpTable.initializeStateAction();
    }
}
