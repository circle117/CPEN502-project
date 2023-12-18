package ece.cpen502.robot;

import ece.cpen502.robot.microBot.MicroBot;
import robocode.ScannedRobotEvent;

public class State {

    public enum EnergyLabel {
        LOW, MEDIUM, HIGH
    }
    public enum DistanceLabel {
        VERY_CLOSE, NEAR, FAR
    }

    private static final int DISTANCE_MIN = 0;
    private static final int DISTANCE_MAX = 1000;
    private static final int ENERGY_MIN = 0;
    private static final int ENERGY_MAX = 100;

    /**
     * map state values to state index
     */
    public double[] indexForLUT(MicroBot robot, ScannedRobotEvent event) {
        double[] index = new double[4];
        index[0] = indexForEnergy(robot.getEnergy());
        index[1] = indexForDistance(event.getDistance());
        index[2] = indexForEnergy(robot.getEnemy().getEnergy());
        double distance = Math.sqrt(Math.pow(robot.getBattleFieldWidth()/2 - robot.getX(), 2) +
                Math.pow(robot.getBattleFieldHeight()/2 - robot.getY(), 2));
        index[3] = indexForDistance(distance);
        return index;
    }

    public double[] inputForNN(MicroBot robot, ScannedRobotEvent event) {
        double[] input = new double[4];
        input[0] = EnergyMinMaxScaling(robot.getEnergy());
        input[1] = DistanceMinMaxScaling(event.getDistance());
        input[2] = EnergyMinMaxScaling(robot.getEnemy().getEnergy());
        double distance = Math.sqrt(Math.pow(robot.getBattleFieldWidth()/2 - robot.getX(), 2) +
                Math.pow(robot.getBattleFieldHeight()/2 - robot.getY(), 2));
        input[3] = DistanceMinMaxScaling(distance);
        return input;
    }

    /**
     * map energy Enum to index
     */
    public int indexForEnergy(double energy) {
        EnergyLabel label = energyTransfer(energy);
        if (label == EnergyLabel.LOW) return 0;
        else if (label == EnergyLabel.MEDIUM) return 1;
        else if (label == EnergyLabel.HIGH) return 2;
        else return -1;
    }

    /**
     * map distance Enum to index
     */
    public int indexForDistance(double distance) {
        DistanceLabel label = distanceTransfer(distance);
        if (label == DistanceLabel.VERY_CLOSE) return 0;
        else if (label == DistanceLabel.NEAR) return 1;
        else if (label == DistanceLabel.FAR) return 2;
        else return -1;
    }

    /**
     * map index to energy Enum
     */
    public EnergyLabel energyTransfer(double energy) {
        if (energy < 33.3) return EnergyLabel.LOW;
        else if (energy < 66.7) return EnergyLabel.MEDIUM;
        else return EnergyLabel.HIGH;
    }

    /**
     * map index to distance Enum
     */
    public DistanceLabel distanceTransfer(double distance) {
        if (distance < 100) return DistanceLabel.VERY_CLOSE;
        else if (distance < 300) return DistanceLabel.NEAR;
        else return DistanceLabel.FAR;
    }

    /**
     * min max scaling for distance
     */
    public double DistanceMinMaxScaling(double distance) {
        return (distance - DISTANCE_MIN)/((DISTANCE_MAX - DISTANCE_MIN) * 1.0);
    }

    /**
     * min max scaling for energy
     */
    public double EnergyMinMaxScaling(double energy) {
        return (energy - ENERGY_MIN)/((ENERGY_MAX - ENERGY_MIN) * 1.0);
    }

    /**
     * map index to distance value
     */
    public double DistanceTransferBack(double index) {
        if (index == 0.0) return Math.random() * 100;
        else if (index == 1.0) return Math.random() * 200 + 100;
        else return Math.random() * 700 + 300;
    }

    /**
     * map index to energy value
     */
    public double EnergyTransferBack(double index) {
        if (index == 0.0) return Math.random() * 33.3;
        else if (index == 1.0) return Math.random() * 33.3 + 33.3;
        else return Math.random() * 33.3 + 66.6;
    }
}
